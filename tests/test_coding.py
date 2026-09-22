"""编码链路（Coder / Tester）的编排测试。

这里刻意**不测**「模型写的代码对不对」—— 那是模型的事，也没法用单测断言。
测的是编排本身：什么时候走编码链路、测试文件有没有算进交付物、
产物重复时会不会出现两条，这些是代码保证得了、也最容易被改坏的部分。
"""
import pytest

from agents import worker
from app.config import settings


@pytest.fixture
def coding_env(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "workspace_dir", str(tmp_path / "ws"))
    monkeypatch.setattr(settings, "code_execution_enabled", True)
    # use_mock 是只读 property（由 mock_llm 与是否配了 Key 推导得出），
    # 不能直接 setattr，只能改它的两个来源
    monkeypatch.setattr(settings, "mock_llm", False)
    monkeypatch.setattr(settings, "deepseek_api_key", "sk-test-not-a-real-key")
    return tmp_path


# ---------------------------------------------------------------- 链路选择
def test_code_task_needs_all_three_conditions(coding_env, monkeypatch):
    subtask = {"id": "t1", "needs_code": True}

    assert worker._is_code_task(subtask) is True

    monkeypatch.setattr(settings, "code_execution_enabled", False)
    assert worker._is_code_task(subtask) is False, "关掉执行开关后不该再走编码链路"


def test_mock_mode_never_goes_to_coding(coding_env, monkeypatch):
    """MockLLM 不支持 function calling，也写不出真代码。

    让它走编码链路只会让离线评估和 CI 变得不稳定 —— 收益为零、噪音拉满。
    """
    monkeypatch.setattr(settings, "mock_llm", True)

    assert worker._is_code_task({"id": "t1", "needs_code": True}) is False


def test_missing_api_key_also_counts_as_mock(coding_env, monkeypatch):
    """没配 Key 时 use_mock 自动为真，同样不该走编码链路。"""
    monkeypatch.setattr(settings, "deepseek_api_key", "")

    assert worker._is_code_task({"id": "t1", "needs_code": True}) is False


def test_plain_task_is_not_code(coding_env):
    """绝大多数子任务是写作/汇总，不该被拉进编码链路白烧几轮 token。"""
    assert worker._is_code_task({"id": "t1"}) is False
    assert worker._is_code_task({"id": "t1", "needs_code": False}) is False


# ---------------------------------------------------------------- 产物合并
def test_code_subtask_merges_test_files_into_artifacts(coding_env, monkeypatch):
    """测试文件也是交付物。

    不把它带进产物，Checker 的验收材料里就只有实现说明，
    没法判断「到底测了什么、测没测边界」。
    """
    monkeypatch.setattr(
        "agents.tester.write_tests",
        lambda subtask, context: ("写了 3 个用例，覆盖空输入", ["tests/test_x.py"]),
    )
    monkeypatch.setattr(
        "agents.coder.implement",
        lambda subtask, context, test_text: ("实现完成，测试全绿", ["solution.py"]),
    )

    output, artifacts = worker._run_code_subtask({"id": "t1"}, "上下文")

    assert artifacts == ["solution.py", "tests/test_x.py"]
    assert "3 个用例" in output
    assert "实现完成" in output


def test_code_subtask_deduplicates_artifacts(coding_env, monkeypatch):
    """Coder 常常会顺手重写一遍测试文件，产物列表不能因此出现两条一样的路径。"""
    monkeypatch.setattr(
        "agents.tester.write_tests",
        lambda subtask, context: ("t", ["tests/test_x.py"]),
    )
    monkeypatch.setattr(
        "agents.coder.implement",
        lambda subtask, context, test_text: ("i", ["solution.py", "tests/test_x.py"]),
    )

    _, artifacts = worker._run_code_subtask({"id": "t1"}, "")

    assert artifacts == ["solution.py", "tests/test_x.py"]


def test_code_subtask_survives_tester_producing_nothing(coding_env, monkeypatch):
    """Tester 一个文件都没写出来时不能崩 —— 交给 Coder 自行判断边界。"""
    monkeypatch.setattr("agents.tester.write_tests", lambda subtask, context: ("", []))
    monkeypatch.setattr(
        "agents.coder.implement",
        lambda subtask, context, test_text: ("我自己判断了边界", ["solution.py"]),
    )

    output, artifacts = worker._run_code_subtask({"id": "t1"}, "")

    assert artifacts == ["solution.py"]
    assert "Tester 未产出测试" in output
