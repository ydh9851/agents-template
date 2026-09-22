"""代码沙箱测试。

这是整个项目里最该「宁可多测」的一块 —— 它会真的把代码跑起来。
下面每个用例都对应一种真实的越界或失效情形，尤其是环境变量那条：
项目 .env 里放着 DEEPSEEK_API_KEY，泄漏出去就等于把钥匙递出去了。
"""
import time
from pathlib import Path

import pytest

from app import sandbox, workspace
from app.config import settings


@pytest.fixture
def sandbox_env(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "workspace_dir", str(tmp_path / "ws"))
    monkeypatch.setattr(settings, "sandbox_backend", "subprocess")
    monkeypatch.setattr(settings, "sandbox_timeout", 10)
    monkeypatch.setattr(settings, "sandbox_output_limit", 4000)
    workspace.workspace_root()  # 先建出目录
    return tmp_path


# ---------------------------------------------------------------- 正常执行
def test_runs_python_and_captures_stdout(sandbox_env):
    workspace.write_text("hello.py", 'print("你好，沙箱")')

    result = sandbox.run_python("hello.py")

    assert result.ok is True
    assert result.exit_code == 0
    assert "你好，沙箱" in result.stdout
    assert result.backend == "subprocess"


def test_passes_argv_to_script(sandbox_env):
    workspace.write_text("args.py", "import sys\nprint(sys.argv[1:])")

    result = sandbox.run_python("args.py", ["a", "b"])

    assert result.ok is True
    assert "['a', 'b']" in result.stdout


def test_workdir_is_locked_to_workspace(sandbox_env):
    """子进程的工作目录必须锁在工作区，否则相对路径会指到宿主目录去。"""
    workspace.write_text("cwd.py", "import os\nprint(os.getcwd())")

    result = sandbox.run_python("cwd.py")

    assert result.ok is True
    assert Path(result.stdout.strip()).resolve() == workspace.workspace_root().resolve()


# ---------------------------------------------------------------- 失败上报
def test_nonzero_exit_keeps_stderr(sandbox_env):
    """失败时 stderr 必须原样带回 —— Coder 就靠 traceback 定位问题。"""
    workspace.write_text("boom.py", "raise ValueError('故意炸的')")

    result = sandbox.run_python("boom.py")

    assert result.ok is False
    assert result.exit_code != 0
    assert "ValueError" in result.stderr
    assert "故意炸的" in result.stderr


def test_result_text_contains_both_streams(sandbox_env):
    workspace.write_text("mix.py", "import sys\nprint('out')\nprint('err', file=sys.stderr)")

    text = sandbox.run_python("mix.py").to_text()

    assert "out" in text and "err" in text
    assert "stdout" in text and "stderr" in text


# ---------------------------------------------------------------- 路径边界
def test_rejects_non_python_file(sandbox_env):
    workspace.write_text("note.md", "# 不是代码")

    result = sandbox.run_python("note.md")

    assert result.ok is False
    assert "只能执行 .py" in result.stderr


def test_rejects_path_escape(sandbox_env):
    """沙箱的第一道关不是沙箱本身，而是工作区 —— 越界必须在更早一层就被拦住。"""
    result = sandbox.run_python("../outside.py")

    assert result.ok is False
    assert "路径不合法" in result.stderr


def test_rejects_missing_file(sandbox_env):
    result = sandbox.run_python("nope.py")

    assert result.ok is False
    assert "不存在" in result.stderr


# ---------------------------------------------------------------- 环境隔离
LEAK_SCRIPT = (
    "import os\n"
    "print('KEY=', os.environ.get('DEEPSEEK_API_KEY', 'MISSING'))\n"
    "print('PWD=', os.environ.get('MYSQL_PASSWORD', 'MISSING'))\n"
    "print('SANDBOX=', os.environ.get('SANDBOX', 'MISSING'))\n"
)


def test_does_not_leak_parent_secrets(sandbox_env, monkeypatch):
    """父进程的密钥绝不能进子进程。

    项目 .env 里放着 DEEPSEEK_API_KEY，一旦泄漏，
    被执行的代码只要 `import os; print(os.environ)` 就能读走。
    这是本模块最不起眼、也最值钱的一条约束。
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-should-never-leak")
    monkeypatch.setenv("MYSQL_PASSWORD", "hunter2")

    workspace.write_text("leak.py", LEAK_SCRIPT)
    result = sandbox.run_python("leak.py")

    assert result.ok is True
    assert "sk-should-never-leak" not in result.stdout
    assert "hunter2" not in result.stdout
    assert "KEY= MISSING" in result.stdout
    assert "PWD= MISSING" in result.stdout


def test_marks_sandbox_flag(sandbox_env):
    workspace.write_text("mark.py", LEAK_SCRIPT)

    assert "SANDBOX= 1" in sandbox.run_python("mark.py").stdout


def test_clean_env_drops_secrets_and_keeps_path(monkeypatch):
    """直接测清洗函数，比隔着子进程断言更直白。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")

    env = sandbox.clean_env()

    assert "DEEPSEEK_API_KEY" not in env
    assert "PATH" in env
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["PYTHONPATH"] == "."


def test_clean_env_is_case_insensitive(monkeypatch):
    """环境变量名大小写在不同平台上不一致，白名单比对必须忽略大小写。"""
    monkeypatch.setenv("deepseek_api_key", "sk-lower")

    assert "deepseek_api_key" not in sandbox.clean_env()


# ---------------------------------------------------------------- 超时
def test_timeout_kills_the_process(sandbox_env, monkeypatch):
    monkeypatch.setattr(settings, "sandbox_timeout", 2)
    workspace.write_text("sleepy.py", "import time\ntime.sleep(30)")

    started = time.perf_counter()
    result = sandbox.run_python("sleepy.py")
    elapsed = time.perf_counter() - started

    assert result.ok is False
    assert result.timed_out is True
    assert elapsed < 15, f"超时未生效，实际耗时 {elapsed:.1f}s"


def test_timeout_is_mentioned_in_text(sandbox_env, monkeypatch):
    monkeypatch.setattr(settings, "sandbox_timeout", 1)
    workspace.write_text("sleepy.py", "import time\ntime.sleep(30)")

    assert "强制终止" in sandbox.run_python("sleepy.py").to_text()


# ---------------------------------------------------------------- 输出截断
def test_output_is_truncated(sandbox_env, monkeypatch):
    """一句 print 就能刷出几 MB —— 不截断的话内存和 LLM 上下文都受不了。"""
    monkeypatch.setattr(settings, "sandbox_output_limit", 200)
    workspace.write_text("loud.py", "print('X' * 5000)")

    result = sandbox.run_python("loud.py")

    assert len(result.stdout) < 600
    assert "已省略" in result.stdout


def test_truncation_keeps_the_tail(sandbox_env, monkeypatch):
    """保尾部而不是头部：traceback 的最后几行才是有用的信息。"""
    monkeypatch.setattr(settings, "sandbox_output_limit", 100)

    text = sandbox._truncate("A" * 1000 + "Traceback: 关键报错")

    assert "Traceback: 关键报错" in text
    assert "已省略" in text


# ---------------------------------------------------------------- pytest
def test_pytest_runs_and_passes(sandbox_env):
    workspace.write_text("solution.py", "def add(a, b):\n    return a + b\n")
    workspace.write_text(
        "tests/test_add.py",
        "from solution import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
    )

    result = sandbox.run_pytest("tests")

    assert result.ok is True, result.to_text()
    assert "1 passed" in result.stdout


def test_pytest_reports_failure(sandbox_env):
    workspace.write_text("solution.py", "def add(a, b):\n    return a - b\n")  # 故意写错
    workspace.write_text(
        "tests/test_add.py",
        "from solution import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
    )

    result = sandbox.run_pytest("tests")

    assert result.ok is False
    assert "assert" in (result.stdout + result.stderr)


def test_pytest_missing_dir_is_reported(sandbox_env):
    """没写测试就跑 pytest 时，要给一句人话，而不是抛异常或一段裸 traceback。"""
    result = sandbox.run_pytest("tests")

    assert result.ok is False
    assert "测试目录不存在" in result.stderr


def test_pytest_rejects_path_escape(sandbox_env):
    result = sandbox.run_pytest("../")

    assert result.ok is False
