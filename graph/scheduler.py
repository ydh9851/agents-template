"""子任务调度：从依赖图里挑出「当前就绪」的子任务。

为什么需要独立一层：
    原先 Worker 靠 `current_index` 线性推进，`deps` 只用于两件事 ——
    把依赖结果喂进 Prompt、以及依赖失败时短路 —— 并没有真正参与调度。
    结果是：一个不依赖任何人的子任务，明明可以立刻做，却必须排在别人后面；
    依赖失败的分支也只能等到被轮到时才发现「输入不完整」。

这一层的判定很朴素，但它是 DAG 调度的核心：

    ready(sub) = 所有依赖都已「通过验收」 且 自己还没完成

一次取出**整层就绪任务**交给 Worker 并行执行，这是相对线性推进最直接的收益 ——
三个互不依赖的子任务从「串行 3 轮」变成「并行 1 轮」。

出边状态只有三种：
    run      有就绪任务，交给 Worker 执行
    blocked  还有未完成的任务，但没有一个就绪 —— 说明它们被失败的依赖堵死了
    idle     全部完成（或压根没拆出子任务）
"""
from __future__ import annotations

from dataclasses import dataclass, field

from graph.state import AgentState


@dataclass
class Schedule:
    """一次调度决策。"""

    action: str  # run / blocked / idle
    # 就绪子任务在 subtasks 里的下标（保持原有顺序，便于日志与结果对齐）
    ready: list[int] = field(default_factory=list)
    # 被失败依赖堵住的子任务 id（action == "blocked" 时才有意义）
    blocked_ids: list[str] = field(default_factory=list)


def verdict_of(state: AgentState) -> dict[str, bool]:
    """子任务 id -> 是否通过验收。只有出现在这里才算「已完成」。"""
    return {str(item.get("id")): bool(item.get("passed")) for item in (state.get("results") or [])}


def schedule(state: AgentState) -> Schedule:
    """计算当前该做什么。"""
    subtasks = state.get("subtasks") or []
    if not subtasks:
        return Schedule(action="idle")

    verdict = verdict_of(state)
    pending = [sub for sub in subtasks if str(sub.get("id")) not in verdict]
    if not pending:
        return Schedule(action="idle")

    ready = [
        index
        for index, sub in enumerate(subtasks)
        if str(sub.get("id")) not in verdict
        # 依赖必须「已通过」才算满足；未完成或已失败都不算
        and all(verdict.get(str(dep)) is True for dep in (sub.get("deps") or []))
    ]
    if ready:
        return Schedule(action="run", ready=ready)

    # 还有没做完的，却一个都就绪 —— 只能是依赖失败把它们堵死了
    return Schedule(action="blocked", blocked_ids=[str(sub.get("id")) for sub in pending])


def pending_ids(state: AgentState) -> list[str]:
    """还没完成验收的子任务 id（供日志使用）。"""
    verdict = verdict_of(state)
    return [str(sub.get("id")) for sub in (state.get("subtasks") or []) if str(sub.get("id")) not in verdict]
