"""LangGraph 状态定义。

整个协作过程只围绕这一个 State 流转：

    Manager  写入 subtasks
    Worker   按「依赖就绪集」并行执行一批，写入 pending_results
    Checker  逐个验收后写入 results（并记录打回理由）
    调度器    根据 results 计算下一批就绪任务

注意这里没有 `current_index` 之类的游标字段：谁该执行由 `graph/scheduler.py`
根据 `subtasks` + `results` 现算，而不是靠一个线性下标推进。
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
    # 本子任务在工作区里产出的文件（相对路径）
    artifacts: list[str]


class PendingResult(TypedDict, total=False):
    """Worker 执行完、等待 Checker 验收的产出。

    为什么不直接写进 results：results 只存「已验收」的结果，
    而并行执行之后要先攒一批产出，等 Checker 逐个判定完才知道该不该入库。
    """

    id: str
    title: str
    output: str
    # Worker 通过工具写入工作区的文件列表
    artifacts: list[str]


class AgentState(TypedDict, total=False):
    """全局状态（会被 SqliteSaver 持久化到 checkpoint）。"""

    # 用户原始任务
    task: str
    # Manager 拆解出的子任务列表
    subtasks: list[SubTask]
    # Worker 本轮并行执行完、待验收的产出
    pending_results: list[PendingResult]
    # 已验收完成的子任务结果
    results: list[SubTaskResult]
    # 每个子任务被打回的重试次数 {subtask_id: count}
    retries: dict[str, int]
    # 最近一次被打回的理由 {subtask_id: reason}，Worker 重做时会带上
    last_rejection: dict[str, str]
    # 本轮 Worker 产出的拼接文本（保留字段：便于 CLI / 日志一眼看清这轮做了什么）
    current_result: str
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
        "pending_results": [],
        "results": [],
        "retries": {},
        "last_rejection": {},
        "current_result": "",
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

    调度器已经保证只把依赖通过的任务交出去，所以正常情况下这里返回空。
    保留它是为了兜底：万一某条路径绕过了调度，也不至于拿失败的依赖继续往下做。
    """
    deps = subtask.get("deps") or []
    if not deps:
        return []
    verdict = {item.get("id"): bool(item.get("passed")) for item in (state.get("results") or [])}
    return [dep for dep in deps if verdict.get(dep) is False]
