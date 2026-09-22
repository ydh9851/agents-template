"""LangGraph 状态机定义 + 运行入口。

图结构：

    manager ──> worker ──> checker ──┬──(未通过)──> worker
                                     └──(通过/超限)──> ... ──> summarize ──> END

配合 checkpointer，每个节点执行完都会自动存一次 SQLite，因此支持断点续跑。
"""
import time
import uuid
from typing import Callable, Iterator

from langgraph.graph import END, StateGraph

from agents.checker import checker_node
from agents.manager import manager_node, summarize_node
from agents.worker import worker_node
from app.llm import get_token_usage, reset_token_usage
from app.observability import bind_thread, get_logger, metrics, node_scope
from app.utils import to_jsonable
from checkpoint.sqlite_checkpoint import open_checkpointer
from graph.scheduler import schedule
from graph.state import AgentState, initial_state

logger = get_logger("graph.workflow")

NODE_MANAGER = "manager"
NODE_WORKER = "worker"
NODE_CHECKER = "checker"
NODE_SUMMARIZE = "summarize"


def _traced(name: str, node: Callable[[AgentState], dict]) -> Callable[[AgentState], dict]:
    """给节点套一层「执行范围」：这个节点内产生的所有日志自动带上 node 标签，并记录耗时。

    为什么不在节点函数里自己打标签：四个节点要重复写四遍，还容易漏。
    放在编排层包一层，节点函数只管业务，标签由「谁调用的」决定。
    """

    def wrapper(state: AgentState) -> dict:
        with node_scope(name):
            logger.info("节点开始：%s", name)
            started = time.perf_counter()
            result = node(state)
            logger.info("节点完成：%s（耗时 %.2fs）", name, time.perf_counter() - started)
            return result

    wrapper.__name__ = f"traced_{name}"
    return wrapper


# ---------- 条件路由 ----------
def route_after_manager(state: AgentState) -> str:
    """没有拆出子任务时直接去汇总，避免空转。"""
    subtasks = state.get("subtasks") or []
    return NODE_WORKER if subtasks else NODE_SUMMARIZE


def route_after_checker(state: AgentState) -> str:
    """还有「没做完的」就继续调度，全部处理完才去汇总。

    判定交给调度器：只要还有未完成的子任务（包括刚被打回、需要重做的），
    就回 Worker 继续；被失败依赖堵住的那些会在 Worker 里一次性标记掉，不会死循环。
    """
    return NODE_WORKER if schedule(state).action != "idle" else NODE_SUMMARIZE


def build_graph(checkpointer=None):
    """构建并编译状态机。checkpointer 传入 SqliteSaver 即可获得断点续跑能力。"""
    builder = StateGraph(AgentState)

    builder.add_node(NODE_MANAGER, _traced(NODE_MANAGER, manager_node))
    builder.add_node(NODE_WORKER, _traced(NODE_WORKER, worker_node))
    builder.add_node(NODE_CHECKER, _traced(NODE_CHECKER, checker_node))
    builder.add_node(NODE_SUMMARIZE, _traced(NODE_SUMMARIZE, summarize_node))

    builder.set_entry_point(NODE_MANAGER)
    builder.add_conditional_edges(
        NODE_MANAGER,
        route_after_manager,
        {NODE_WORKER: NODE_WORKER, NODE_SUMMARIZE: NODE_SUMMARIZE},
    )
    builder.add_edge(NODE_WORKER, NODE_CHECKER)
    builder.add_conditional_edges(
        NODE_CHECKER,
        route_after_checker,
        {NODE_WORKER: NODE_WORKER, NODE_SUMMARIZE: NODE_SUMMARIZE},
    )
    builder.add_edge(NODE_SUMMARIZE, END)

    return builder.compile(checkpointer=checkpointer)


def _build_config(thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        # 每次重试都会多跑一轮节点，这里放宽递归上限
        "recursion_limit": 100,
    }


# ---------- 运行入口 ----------
def stream_task(task: str | None = None, thread_id: str | None = None, resume: bool = False) -> Iterator[tuple[str, dict]]:
    """流式执行任务，逐个节点 yield (事件名, 数据)。

    事件类型：start / node / done / error
    """
    thread_id = thread_id or uuid.uuid4().hex
    config = _build_config(thread_id)

    # 每次执行都重置 token 计量并绑定上下文：
    # 并发跑多个任务时，各自的日志和用量必须互不干扰（contextvar 天然按上下文隔离）
    reset_token_usage()
    bind_thread(thread_id)
    started = time.perf_counter()

    with open_checkpointer() as checkpointer:
        graph = build_graph(checkpointer)

        if resume:
            snapshot = graph.get_state(config)
            if not snapshot.values:
                raise ValueError(f"thread_id={thread_id} 没有可恢复的 checkpoint")
            payload = None
        else:
            if not task:
                raise ValueError("首次执行任务时必须提供 task")
            payload = initial_state(task)

        yield "start", to_jsonable(
            {"thread_id": thread_id, "resume": resume, "task": task or (payload or {}).get("task", "")}
        )

        try:
            for chunk in graph.stream(payload, config, stream_mode="updates"):
                for node, update in chunk.items():
                    yield "node", to_jsonable({"node": node, "update": update})

            final_state = dict(graph.get_state(config).values or {})
            elapsed = time.perf_counter() - started
            tokens = get_token_usage()
            status = str(final_state.get("status") or "unknown")

            metrics.inc("tasks_total", status=status)
            metrics.observe("task_duration_seconds", elapsed, status=status)
            metrics.inc("task_tokens_total", tokens)
            logger.info("任务结束：status=%s，耗时 %.2fs，tokens=%d，thread=%s",
                        status, elapsed, tokens, thread_id)

            # token 用量与总耗时随 done 事件返回：前端不额外查询就能展示本次任务的成本
            final_state.update({"token_usage": tokens, "elapsed_ms": int(elapsed * 1000)})
            yield "done", to_jsonable(final_state)
        except Exception as exc:  # noqa: BLE001 - 把异常也通过 SSE 回传，方便前端提示
            metrics.inc("tasks_total", status="error")
            metrics.observe("task_duration_seconds", time.perf_counter() - started, status="error")
            logger.exception("任务执行失败")
            yield "error", to_jsonable({"message": str(exc)})


def run_task(task: str | None = None, thread_id: str | None = None, resume: bool = False) -> dict:
    """同步执行任务，返回最终状态（供 /task 与 CLI 使用）。"""
    final_state: dict = {}
    error: str | None = None
    for event, data in stream_task(task=task, thread_id=thread_id, resume=resume):
        if event == "done":
            final_state = data
        elif event == "error":
            error = data.get("message", "未知错误")
    if error and not final_state:
        raise RuntimeError(error)
    return final_state


def get_task_state(thread_id: str) -> dict:
    """查询某个 thread 的最新状态（用于验证断点是否落盘）。"""
    config = _build_config(thread_id)
    with open_checkpointer() as checkpointer:
        graph = build_graph(checkpointer)
        snapshot = graph.get_state(config)
        return {
            "thread_id": thread_id,
            "next": list(snapshot.next or []),
            "values": to_jsonable(snapshot.values) if snapshot.values else {},
        }
