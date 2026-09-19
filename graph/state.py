"""LangGraph 状态定义。

整个协作过程只围绕这一个 State 流转：
Manager 写入 subtasks，Worker 写入 current_result，Checker 写入 results 并推进 current_index。
"""
from typing import Any, TypedDict


class SubTask(TypedDict, total=False):
    """Manager 拆出来的子任务。"""

    id: str
    title: str
    description: str
    deps: list[str]
    acceptance: str
    need_rag: bool


class SubTaskResult(TypedDict, total=False):
    """Checker 验收后落库的子任务结果。"""

    id: str
    title: str
    output: str
    passed: bool
    reason: str
    attempts: int


class AgentState(TypedDict, total=False):
    """全局状态（会被 SqliteSaver 持久化到 checkpoint）。"""

    # 用户原始任务
    task: str
    # Manager 拆解出的子任务列表
    subtasks: list[SubTask]
    # 当前正在处理的子任务下标
    current_index: int
    # Worker 当前轮次的产出
    current_result: str
    # 已验收完成的子任务结果
    results: list[SubTaskResult]
    # 每个子任务被打回的重试次数 {subtask_id: count}
    retries: dict[str, int]
    # 可读的执行日志（便于排查与前端展示）
    logs: list[str]
    # 最终汇总结果
    final_answer: str
    # running / finished / partial
    status: str


def initial_state(task: str) -> AgentState:
    """构造一次新任务的初始状态。"""
    return {
        "task": task,
        "subtasks": [],
        "current_index": 0,
        "current_result": "",
        "results": [],
        "retries": {},
        "logs": [],
        "final_answer": "",
        "status": "running",
    }


def append_log(state: AgentState, message: str) -> list[str]:
    """返回追加日志后的新列表（State 里 list 是整体覆盖，所以每次都要返回全量）。"""
    logs: list[Any] = list(state.get("logs") or [])
    logs.append(message)
    return logs


def failed_deps(state: AgentState, subtask: SubTask) -> list[str]:
    """返回该子任务中「已验收但未通过」的依赖 id。

    用于依赖失败时的快速短路：前置子任务没通过，后续子任务缺少有效输入，
    再让模型重试也只会产出「无法完成」的声明，白白消耗 token。
    """
    deps = subtask.get("deps") or []
    if not deps:
        return []
    verdict = {item.get("id"): bool(item.get("passed")) for item in (state.get("results") or [])}
    return [dep for dep in deps if verdict.get(dep) is False]
