"""Manager Agent：把复杂任务拆成子任务列表；全部完成后负责汇总。

对应 LangGraph 里的两个节点：
- manager_node：任务拆解（入口节点）
- summarize_node：结果汇总（出口节点）
"""
from app.config import settings
from app.llm import chat
from app.prompts import load_prompt
from app.utils import extract_json, get_logger, render
from graph.state import AgentState, append_log, initial_state

logger = get_logger("agent.manager")


def _clean_text(value: object, default: str = "") -> str:
    return str(value).strip() if value is not None else default


def normalize_subtasks(raw_subtasks: object) -> list[dict]:
    """清洗模型返回的子任务：补字段、去重复 id、过滤非法依赖、限制数量。"""
    if not isinstance(raw_subtasks, list):
        raise ValueError("模型返回的 subtasks 不是列表")

    cleaned: list[dict] = []
    used_ids: set[str] = set()
    for index, item in enumerate(raw_subtasks):
        if not isinstance(item, dict):
            continue
        sub_id = _clean_text(item.get("id")) or f"t{index + 1}"
        # id 去重，避免后续 retries 字典互相覆盖
        while sub_id in used_ids:
            sub_id = f"{sub_id}_x"
        used_ids.add(sub_id)

        deps = item.get("deps") or []
        if not isinstance(deps, list):
            deps = []
        # 依赖只能指向已出现过的子任务
        valid_deps = [str(dep) for dep in deps if str(dep) in used_ids and str(dep) != sub_id]

        cleaned.append(
            {
                "id": sub_id,
                "title": _clean_text(item.get("title"), f"子任务 {sub_id}"),
                "description": _clean_text(item.get("description"), _clean_text(item.get("title"))),
                "deps": valid_deps,
                "acceptance": _clean_text(item.get("acceptance"), "结果清晰、完整、可交付"),
                "need_rag": bool(item.get("need_rag", True)),
                # 代码类子任务会走「Tester 出题 → Coder 实现并跑到通过」的独立链路。
                # 默认 false：绝大多数子任务是写作/汇总，不该被拉进编码链路白白多花几轮 token。
                "needs_code": bool(item.get("needs_code", False)),
            }
        )

    if not cleaned:
        raise ValueError("模型没有拆出任何有效子任务")
    if len(cleaned) > settings.max_subtasks:
        logger.warning("子任务数量超过上限 %d，已截断", settings.max_subtasks)
        cleaned = cleaned[: settings.max_subtasks]
    return cleaned


def split_task(task: str) -> list[dict]:
    """把用户任务拆成（清洗后的）子任务列表。

    抽成独立函数是为了让离线评估（evals/run_eval.py）跑的就是**线上同一套
    Prompt 与清洗逻辑** —— 评估脚本自己另拼一份平行实现的话，
    测出来的分数代表不了线上行为，等于白测。
    """
    system_prompt = render(load_prompt("manager"), max_subtasks=settings.max_subtasks)
    raw = chat(system_prompt, f"用户任务：\n{task}")
    return normalize_subtasks(extract_json(raw).get("subtasks"))


def manager_node(state: AgentState) -> dict:
    """入口节点：拆任务。"""
    task = state.get("task", "")
    logger.info("Manager 开始拆解任务：%s", task[:60])

    subtasks = split_task(task)

    titles = "、".join(sub["title"] for sub in subtasks)
    return {
        "subtasks": subtasks,
        "pending_results": [],
        "results": [],
        "retries": {},
        "last_rejection": {},
        "current_result": "",
        "final_answer": "",
        "status": "running",
        "logs": append_log(state, f"[Manager] 拆解出 {len(subtasks)} 个子任务：{titles}"),
    }


def summarize_node(state: AgentState) -> dict:
    """出口节点：把所有子任务结果汇总成最终交付物。"""
    task = state.get("task", "")
    results = state.get("results") or []

    blocks: list[str] = []
    for item in results:
        flag = "已通过" if item.get("passed") else "未通过"
        blocks.append(
            f"### {item.get('id')} {item.get('title')}（{flag}，尝试 {item.get('attempts', 1)} 次）\n"
            f"验收意见：{item.get('reason', '')}\n\n{item.get('output', '')}"
        )
    payload = f"用户任务：\n{task}\n\n各子任务结果：\n" + "\n\n".join(blocks)

    logger.info("Manager 开始汇总 %d 个子任务结果", len(results))
    final_answer = chat(load_prompt("summarizer"), payload)

    failed = [item.get("id") for item in results if not item.get("passed")]
    status = "partial" if failed else "finished"
    log = f"[Manager] 汇总完成，状态={status}"
    if failed:
        log += f"，未通过验收的子任务：{', '.join(str(x) for x in failed)}"

    return {
        "final_answer": final_answer,
        "status": status,
        "logs": append_log(state, log),
    }


__all__ = ["manager_node", "summarize_node", "normalize_subtasks", "initial_state"]
