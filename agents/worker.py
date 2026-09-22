"""Worker Agent：按「依赖就绪集」并行执行子任务，需要资料时调混合检索 RAG。

与旧版的区别：
    旧版靠 `current_index` 线性推进，一次只做一个子任务；
    现在谁该执行由 graph/scheduler.py 根据依赖现算，一次把**整层就绪任务**取出来并行做。
    三个互不依赖的子任务从「串行 3 轮」变成「并行 1 轮」。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from agents.loop import run_with_tools
from app.config import settings
from app.llm import LLMUnavailable, TokenBudgetExceeded, chat
from app.prompts import load_prompt
from app.tools import tools_enabled
from app.utils import get_logger
from graph.scheduler import schedule
from graph.state import AgentState, PendingResult, append_log, failed_deps

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


def _last_rejection(state: AgentState, subtask_id: str) -> str:
    """取该子任务上一次被打回的理由。

    这是「带意见打回重做」真正生效的地方。旧版想从 results 里找理由，
    但被打回的子任务根本不会写进 results（只有验收完的才入库），
    所以那段逻辑实际上从来没拿到过任何理由 —— 打回重做退化成「原样再做一遍」。
    """
    return str((state.get("last_rejection") or {}).get(subtask_id, ""))


def _is_code_task(subtask: dict) -> bool:
    """是否走编码链路。三个条件缺一不可：

      1. Manager 标了 needs_code
      2. 代码执行没被配置关掉（CODE_EXECUTION_ENABLED）
      3. 不是 Mock 模式 —— MockLLM 既不支持 function calling，也写不出真代码
    """
    return bool(
        subtask.get("needs_code")
        and settings.code_execution_enabled
        and not settings.use_mock
    )


def _run_code_subtask(subtask: dict, context: str) -> tuple[str, list[str]]:
    """代码子任务：Tester 先出题，Coder 再实现并跑到测试通过。

    顺序是刻意的（先出题、后实现）：
        如果先有实现，测试会不自觉地照着实现已有的边界去写 ——
        「实现怎么跑，测试就怎么测」，等于没测。
    这正是 Coder / Tester 必须分成两个角色的全部理由。
    """
    from agents.coder import implement
    from agents.tester import write_tests

    test_text, test_files = write_tests(subtask, context)
    logger.info("子任务 %s 走编码链路：Tester 产出 %d 个测试文件", subtask["id"], len(test_files))

    impl_text, artifacts = implement(subtask, context, test_text)

    output = (
        f"{impl_text}\n\n"
        f"——\n"
        f"【Tester 独立编写的测试】\n{test_text or '（Tester 未产出测试）'}\n"
        f"【测试文件】{', '.join(test_files) or '（无）'}"
    )
    # 测试文件也是交付物：Checker 要能看到它，否则没法判断「到底测了什么」
    merged = list(dict.fromkeys([*artifacts, *test_files]))
    return output, merged


def _run_subtask(state: AgentState, subtask: dict) -> PendingResult:
    """执行单个子任务。

    刻意写成「只读 state」的形式：并行执行时多个子任务共享同一个 state，
    只要不往里写就不会互相干扰（写入统一由 worker_node 汇总）。
    """
    retry_count = int((state.get("retries") or {}).get(subtask["id"], 0))

    hits: list[dict] = []
    if subtask.get("need_rag", True):
        query = f"{subtask.get('title', '')} {subtask.get('description', '')}"
        hits = retrieve_context(query)

    context = _format_context(hits)
    user_prompt = (
        f"【当前子任务】\n"
        f"id：{subtask['id']}\n"
        f"标题：{subtask['title']}\n"
        f"描述：{subtask['description']}\n"
        f"验收标准：{subtask['acceptance']}\n\n"
        f"【依赖子任务的已完成结果】\n{_format_deps(state, subtask)}\n\n"
        f"【上一轮验收意见】\n{_last_rejection(state, subtask['id']) or '（首次执行）'}\n\n"
        f"【检索资料】\n{context}\n\n"
        f"请完成该子任务。"
    )

    system_prompt = load_prompt("worker")
    artifacts: list[str] = []
    try:
        if _is_code_task(subtask):
            # 代码子任务：Tester 先出题 → Coder 实现并跑到测试通过
            output, artifacts = _run_code_subtask(subtask, context)
        elif tools_enabled() and not settings.use_mock:
            # 普通子任务：走 ReAct 循环，允许它读工作区、把成品写进去
            output, artifacts = run_with_tools(
                system_prompt, user_prompt, tag=f"worker/{subtask['id']}",
            )
        else:
            # Mock 模式刻意不走工具这条路 —— MockLLM 不支持 function calling，
            # 强行走只会让离线链路和 CI 变得不稳定
            output = chat(system_prompt, user_prompt)
    except (TokenBudgetExceeded, LLMUnavailable) as exc:
        # 模型彻底不可用时不抛异常打断整图：产出一段明确的「未执行」说明，
        # 让 Checker 判它未通过、流程继续往下走，最终 status=partial 如实反映结果。
        logger.warning("子任务 %s 未能执行：%s", subtask["id"], exc)
        output = f"【未执行】{exc}"

    detail = f"，引用 {len(hits)} 条检索片段" if hits else ""
    if artifacts:
        detail += f"，产出 {len(artifacts)} 个文件（{', '.join(artifacts)}）"
    logger.info("子任务 %s 完成（第 %d 次尝试）%s", subtask["id"], retry_count + 1, detail)

    return {
        "id": subtask["id"],
        "title": subtask.get("title", ""),
        "output": output,
        "artifacts": artifacts,
    }


def _mark_blocked(state: AgentState) -> dict:
    """剩余子任务全被失败的依赖堵死：一次性标记为未通过，避免整图空转。"""
    results = list(state.get("results") or [])
    done = {item.get("id") for item in results}
    marked: list[str] = []

    for sub in state.get("subtasks") or []:
        sub_id = sub.get("id")
        if sub_id in done:
            continue
        failed = failed_deps(state, sub)
        results.append(
            {
                "id": sub_id,
                "title": sub.get("title", ""),
                "output": "",
                "passed": False,
                "reason": (
                    f"依赖子任务未通过验收（{', '.join(failed)}），缺少有效输入，未执行。"
                    if failed
                    else "所在依赖链已中断，未执行。"
                ),
                "attempts": 0,
                "artifacts": [],
            }
        )
        marked.append(str(sub_id))

    logger.warning("剩余 %d 个子任务因依赖失败被跳过：%s", len(marked), marked)
    return {
        "results": results,
        "pending_results": [],
        "current_result": "",
        "logs": append_log(state, f"[Worker] 子任务 {', '.join(marked)} 因依赖失败跳过执行"),
    }


def worker_node(state: AgentState) -> dict:
    """执行当前所有就绪子任务（同层并行）。"""
    plan = schedule(state)

    if plan.action == "idle":
        return {
            "pending_results": [],
            "current_result": "",
            "logs": append_log(state, "[Worker] 没有待执行的子任务，跳过"),
        }
    if plan.action == "blocked":
        return _mark_blocked(state)

    subtasks = state.get("subtasks") or []
    batch = [subtasks[index] for index in plan.ready]

    if len(batch) == 1:
        pending = [_run_subtask(state, batch[0])]
    else:
        workers = max(1, min(len(batch), int(settings.max_parallel_subtasks)))
        logger.info("本轮 %d 个子任务就绪，并行执行（并发上限 %d）", len(batch), workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # pool.map 保持输入顺序，产出顺序与 batch 一致，便于日志与结果对齐
            pending = list(pool.map(lambda sub: _run_subtask(state, sub), batch))

    return {
        "pending_results": pending,
        "current_result": "\n\n".join(str(item.get("output", "")) for item in pending),
        "logs": append_log(
            state,
            f"[Worker] 完成子任务 {', '.join(str(item.get('id')) for item in pending)}"
            f"（本轮 {len(pending)} 个）",
        ),
    }
