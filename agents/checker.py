"""Checker Agent：验收 Worker 的结果，判定通过 / 打回重做。

节点里同时负责推进状态：
- 通过：结果入库，current_index + 1；
- 不通过且未超过重试上限：只累加重试次数，current_index 不变，路由会把流程送回 Worker；
- 不通过且超过上限：标记该子任务失败并入库，继续下一个（避免整图死循环）。
"""
from app.config import settings
from app.llm import chat
from app.prompts import load_prompt
from app.utils import extract_json, get_logger
from graph.state import AgentState, append_log, failed_deps

logger = get_logger("agent.checker")


def checker_node(state: AgentState) -> dict:
    """验收当前子任务的产出。"""
    index = int(state.get("current_index", 0))
    subtasks = state.get("subtasks") or []
    if index >= len(subtasks):
        return {"logs": append_log(state, "[Checker] 无待验收子任务，跳过")}

    subtask = subtasks[index]
    output = state.get("current_result", "")
    retries = dict(state.get("retries") or {})
    results = list(state.get("results") or [])
    attempts = int(retries.get(subtask["id"], 0)) + 1

    # 前置依赖未通过时直接标记失败并推进，不调用模型、也不重试：
    # 输入没有变化，重试必然得到同样的「无法完成」，纯属浪费 token。
    blocked = failed_deps(state, subtask)
    if blocked:
        results.append(
            {
                "id": subtask["id"],
                "title": subtask["title"],
                "output": output,
                "passed": False,
                "reason": f"依赖子任务未通过验收（{', '.join(blocked)}），缺少有效输入，跳过执行。",
                "attempts": 1,
            }
        )
        retries.pop(subtask["id"], None)
        return {
            "results": results,
            "retries": retries,
            "current_index": index + 1,
            "current_result": "",
            "logs": append_log(
                state,
                f"[Checker] 子任务 {subtask['id']} 因依赖 {', '.join(blocked)} 失败而跳过验收，标记为未通过",
            ),
        }

    user_prompt = (
        f"【子任务】\n"
        f"id：{subtask['id']}\n"
        f"标题：{subtask['title']}\n"
        f"描述：{subtask['description']}\n"
        f"验收标准：{subtask['acceptance']}\n\n"
        f"【Worker 的结果】\n{output}\n\n"
        f"【验收轮次】该子任务第 {attempts} 次验收（上限 {settings.max_retry_per_subtask} 次重试）\n\n"
        f"请判定该结果是否达到验收标准。"
    )

    try:
        verdict = extract_json(chat(load_prompt("checker"), user_prompt))
        passed = bool(verdict.get("pass"))
        reason = str(verdict.get("reason", "")).strip() or "（模型未给出理由）"
    except Exception as exc:  # noqa: BLE001 - 判定失败时放行，避免任务被卡死
        logger.warning("Checker 判定失败，按通过处理：%s", exc)
        passed, reason = True, f"判定异常，自动放行：{exc}"

    logger.info("Checker 子任务 %s 判定：%s", subtask["id"], "通过" if passed else "不通过")

    # ---- 判定通过 ----
    if passed:
        results.append(
            {
                "id": subtask["id"],
                "title": subtask["title"],
                "output": output,
                "passed": True,
                "reason": reason,
                "attempts": attempts,
            }
        )
        return {
            "results": results,
            "retries": retries,
            "current_index": index + 1,
            "current_result": "",
            "logs": append_log(state, f"[Checker] 子任务 {subtask['id']} 通过验收"),
        }

    # ---- 判定不通过 ----
    retries[subtask["id"]] = attempts
    exceeded = attempts > settings.max_retry_per_subtask

    if exceeded:
        results.append(
            {
                "id": subtask["id"],
                "title": subtask["title"],
                "output": output,
                "passed": False,
                "reason": f"已达最大重试次数（{settings.max_retry_per_subtask}），最后一次验收意见：{reason}",
                "attempts": attempts,
            }
        )
        return {
            "results": results,
            "retries": retries,
            "current_index": index + 1,
            "current_result": "",
            "logs": append_log(
                state,
                f"[Checker] 子任务 {subtask['id']} 连续 {attempts} 次未通过，标记为失败并继续",
            ),
        }

    return {
        "results": results,
        "retries": retries,
        "current_result": output,
        "logs": append_log(
            state,
            f"[Checker] 子任务 {subtask['id']} 未通过（第 {attempts} 次），打回 Worker 重做：{reason[:60]}",
        ),
    }
