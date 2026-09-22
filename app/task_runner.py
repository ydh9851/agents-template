"""后台任务执行 + 事件缓冲：SSE 断线续传的基础设施。

原先 `/task/stream` 是「一个生成器直接把事件透传给客户端」：
    客户端一断线，生成器被关闭，任务随之中止；重连只能从头再跑一遍 ——
    而一个任务要跑几十秒到几分钟，这在实际使用里很难接受。

要做到断线续传，任务必须**脱离连接独立运行**：

    1. 执行放进后台线程，事件先写进缓冲区；
    2. SSE 连接只是缓冲区的一个读者 —— 它断了不影响任务；
    3. 重连时带 `Last-Event-ID`，从缓冲区对应位置继续读。

这样两种「中断」就都有解法了：
    checkpoint  解决「进程崩了怎么办」
    本模块       解决「网断了怎么办」

内存与生命周期：
    事件缓冲放在进程内存里，保留最近若干次执行（TaskRegistry.max_runs）。
    这是刻意的取舍 —— 上 Redis / 消息队列能做得更「生产」，但那是另一个量级的复杂度，
    对这个「一个文件就能读懂」的项目不划算。多副本部署下需要换成共享存储。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator


@dataclass(frozen=True)
class Event:
    """一条可重放的事件。id 单调递增，就是 SSE 的 `Last-Event-ID` 依据。"""

    id: int
    name: str
    data: dict


@dataclass
class TaskRun:
    """一次后台执行 + 它产生的事件流。

    线程模型：后台线程只负责 append，SSE 线程只负责 wait_for / 读取，
    两者通过一把条件变量同步 —— 不需要额外的队列，也不会有忙等。
    """

    thread_id: str
    task: str | None
    resume: bool
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self._cond = threading.Condition()
        self._events: list[Event] = []
        self._status = "running"  # running / finished
        self._worker: threading.Thread | None = None

    # ---------- 写入（后台线程） ----------
    def append(self, name: str, data: dict) -> None:
        with self._cond:
            self._events.append(Event(id=len(self._events), name=name, data=data))
            self._cond.notify_all()

    def mark_finished(self) -> None:
        with self._cond:
            self._status = "finished"
            self._cond.notify_all()

    # ---------- 读取（SSE 线程） ----------
    @property
    def finished(self) -> bool:
        with self._cond:
            return self._status == "finished"

    @property
    def event_count(self) -> int:
        with self._cond:
            return len(self._events)

    def wait_for(self, after_id: int, timeout: float = 1.0) -> list[Event]:
        """返回 id > after_id 的事件。

        没有新事件时阻塞至多 timeout 秒（而不是忙等）：
        任务跑完会 notify，所以正常情况下会被立刻唤醒；
        超时只是为了周期性回到调用方，好让它有机会发心跳、检查客户端是否还在。
        """
        with self._cond:
            if self._status == "running" and len(self._events) <= after_id + 1:
                self._cond.wait(timeout)
            return list(self._events[after_id + 1:])

    # ---------- 执行 ----------
    def start(self, runner: Callable[..., Iterator[tuple[str, dict]]]) -> None:
        self._worker = threading.Thread(
            target=self._run, args=(runner,), daemon=True, name=f"task-{self.thread_id}"
        )
        self._worker.start()

    def _run(self, runner: Callable[..., Iterator[tuple[str, dict]]]) -> None:
        try:
            for name, data in runner(self.task, self.thread_id, self.resume):
                self.append(name, data)
        except Exception as exc:  # noqa: BLE001 - 后台线程里的异常只能变成事件，不能往外抛
            self.append("error", {"message": f"{type(exc).__name__}: {exc}"})
        finally:
            self.mark_finished()


class TaskRegistry:
    """线程安全的任务登记表：thread_id -> TaskRun。"""

    def __init__(self, max_runs: int = 50) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, TaskRun] = {}
        self._max_runs = max_runs

    def get(self, thread_id: str) -> TaskRun | None:
        with self._lock:
            return self._runs.get(thread_id)

    def start(
        self,
        thread_id: str,
        task: str | None,
        resume: bool,
        runner: Callable[..., Iterator[tuple[str, dict]]],
    ) -> TaskRun:
        """启动一次执行；同一个 thread_id 已经有记录时直接复用。

        「复用」正是断线重连的基础：客户端带着同一个 thread_id 回来时，
        拿到的必须是**同一个 TaskRun**，否则就变成重跑而不是续传。

        语义上：一个 thread_id 只对应一次执行 ——
        它还在跑时复用叫「续传」，已经跑完时复用叫「重放」。
        想跑一个全新任务请换一个 thread_id（客户端不传就自动生成）。
        """
        with self._lock:
            existing = self._runs.get(thread_id)
            if existing is not None:
                return existing

            run = TaskRun(thread_id=thread_id, task=task, resume=resume)
            self._runs[thread_id] = run
            self._evict_locked()

        run.start(runner)
        return run

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._runs)

    def _evict_locked(self) -> None:
        """超出容量时优先淘汰最早结束的执行，尽量不丢正在跑的。"""
        if len(self._runs) <= self._max_runs:
            return
        finished = sorted(
            (run for run in self._runs.values() if run.finished),
            key=lambda run: run.created_at,
        )
        for run in finished:
            if len(self._runs) <= self._max_runs:
                break
            self._runs.pop(run.thread_id, None)

    def clear(self) -> None:
        """测试隔离用。"""
        with self._lock:
            self._runs.clear()


task_registry = TaskRegistry()


def parse_last_event_id(raw: str | None) -> int:
    """解析 SSE 的 Last-Event-ID 请求头。非法值一律当成「从头开始」。"""
    if not raw:
        return -1
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return -1
