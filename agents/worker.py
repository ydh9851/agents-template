"""Worker Agent：执行当前子任务，需要查资料时调混合检索 RAG。"""
from app.config import settings
from app.llm import chat
from app.prompts import load_prompt
from app.utils import get_logger, render
from graph.state import AgentState, append_log, failed_deps

logger = get_logger("agent.worker")


def retrieve_context(query: str) -> list[dict]:
    """调混合检索。检索失败不应该中断任务，只降级为「没有资料」。"""
    try:
        from rag.hybrid_search import get_retriever

        return get_retriever().search(query, top_k=settings.rag_top_k)
    except Exception as exc:  # noqa: BLE001 - 检索是增强项，异常一律降级
        logger.warning("混合检索失败，本次将不注入资料：%s", exc)
        return []


def _format_context(hits: list[dict]) -> str:
    """把检索片段拼成带编号的引用块，编号与 Prompt 里要求的 [1] 对应。

    来源精确到 chunk 序号（例如 08-rrf.md#2），避免同一文档的不同片段引用时无法区分。
    """
    if not hits:
        return "（未检索到相关资料，请基于常识作答，并在自检里说明）"
    lines = []
    for index, hit in enumerate(hits, start=1):
        metadata = hit.get("metadata") or {}
        source = hit.get("source") or metadata.get("source", "未知来源")
        chunk_no = metadata.get("chunk")
        label = f"{source}#{chunk_no}" if chunk_no is not None else source
        title = metadata.get("title", "")
        lines.append(
            f"[{index}] 来源：{label}{f'（{title}）' if title else ''}，融合分 {hit.get('score')}\n"
            f"{hit.get('text', '').strip()}"
        )
    return "\n\n".join(lines)


def _format_deps(state: AgentState, subtask: dict) -> str:
    """拼出依赖子任务的已完成结果，让 Worker 能接着往下做。"""
    deps = subtask.get("deps") or []
    if not deps:
        return "（无）"
    results = {item.get("id"): item for item in (state.get("results") or [])}
    blocks = []
    for dep_id in deps:
        item = results.get(dep_id)
        if item:
            blocks.append(f"- {dep_id} {item.get('title', '')}：\n{item.get('output', '')}")
        else:
            blocks.append(f"- {dep_id}：（依赖结果缺失，请自行合理假设）")
    return "\n".join(blocks)


def worker_node(state: AgentState) -> dict:
    """执行当前子任务，产出写回 current_result。"""
    index = int(state.get("current_index", 0))
    subtasks = state.get("subtasks") or []
    if index >= len(subtasks):
        # 理论上不会走到这里，做个兜底避免整图崩溃
        return {"current_result": "", "logs": append_log(state, "[Worker] 无待执行子任务，跳过")}

    subtask = subtasks[index]
    retry_count = int((state.get("retries") or {}).get(subtask["id"], 0))
    logger.info("Worker 执行子任务 %s（第 %d 次尝试）", subtask["id"], retry_count + 1)

    # 前置依赖没通过验收时直接短路，不调用模型（输入不会变，重试只会重复输出「无法完成」）
    blocked = failed_deps(state, subtask)
    if blocked:
        logger.warning("子任务 %s 的依赖 %s 未通过，跳过执行", subtask["id"], blocked)
        return {
            "current_result": (
                f"【阻塞】依赖子任务 {', '.join(blocked)} 未通过验收，缺少可用的输入依据，"
                f"本子任务不执行，以免编造内容。"
            ),
            "logs": append_log(
                state, f"[Worker] 子任务 {subtask['id']} 因依赖 {', '.join(blocked)} 失败而跳过执行"
            ),
        }

    # 需要资料就去调 RAG；上一轮被 Checker 打回时，把打回意见一起喂回去
    hits: list[dict] = []
    if subtask.get("need_rag", True):
        query = f"{subtask.get('title', '')} {subtask.get('description', '')}"
        hits = retrieve_context(query)

    last_reason = ""
    if retry_count > 0:
        for item in reversed(state.get("results") or []):
            if item.get("id") == subtask["id"]:
                last_reason = str(item.get("reason", ""))
                break

    user_prompt = (
        f"【当前子任务】\n"
        f"id：{subtask['id']}\n"
        f"标题：{subtask['title']}\n"
        f"描述：{subtask['description']}\n"
        f"验收标准：{subtask['acceptance']}\n\n"
        f"【依赖子任务的已完成结果】\n{_format_deps(state, subtask)}\n\n"
        f"【上一轮验收意见】\n{last_reason or '（首次执行）'}\n\n"
        f"【检索资料】\n{_format_context(hits)}\n\n"
        f"请完成该子任务。"
    )
    result = chat(load_prompt("worker"), user_prompt)

    log = f"[Worker] 完成子任务 {subtask['id']}（{subtask['title']}）"
    if hits:
        log += f"，引用 {len(hits)} 条检索片段"
    return {
        "current_result": result,
        "logs": append_log(state, log),
    }
