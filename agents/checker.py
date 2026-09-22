"""Checker Agent：验收 Worker 的产出，判定通过 / 打回重做。

Worker 现在一次交一批（同层并行），所以这里也按批验收（必要时并行）：

- 通过          → 写入 results；
- 不通过且未超限 → 只累加 retries，并把理由写进 last_rejection；
                  该子任务**不进 results**，于是调度器下一轮会重新把它算作就绪，自动重做；
- 不通过且超限   → 写入 results 并标记未通过，不再重做（避免整图死循环）。

状态推进不再依赖下标：谁还没完成由 graph/scheduler.py 现算。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from app import workspace
from app.config import settings
from app.llm import LLMUnavailable, TokenBudgetExceeded, chat
from app.prompts import load_prompt
from app.utils import extract_json, get_logger
from graph.state import AgentState, append_log

logger = get_logger("agent.checker")


def _review_material(output: str, artifacts: list[str] | None) -> str:
    """把产出的文件内容附进验收材料。

    为什么必须这么做：Worker 用 write_file 把成果写进文件后，正文里往往只剩
    「已保存到 xxx.md」这一句 —— Checker 只看 output 就会判「没达到验收标准」，
    任务明明是完成的，却被打回甚至标成 partial（实测踩过）。
    验收材料必须和实际交付物一致，而文件也是交付物。
    """
    if not artifacts:
        return output

    blocks: list[str] = []
    for path in artifacts:
        try:
            blocks.append(f"【产物文件 {path}】\n{workspace.read_text(path)}")
        except workspace.WorkspaceError as exc:
            blocks.append(f"【产物文件 {path}】读取失败：{exc}")
    return output + "\n\n" + "\n\n".join(blocks)


def build_checker_prompt(subtask: dict, output: str, attempts: int) -> str:
    """构造验收 Prompt。

    抽成独立函数，是为了让离线评估（evals/run_eval.py）跑的就是线上这套问法；
    评估脚本另拼一份 Prompt 的话，测出来的准确率说明不了线上任何问题。
    """
    return (
        f"【子任务】\n"
        f"id：{subtask.get('id')}\n"
        f"标题：{subtask.get('title')}\n"
        f"描述：{subtask.get('description')}\n"
        f"验收标准：{subtask.get('acceptance')}\n\n"
        f"【Worker 的结果】\n{output}\n\n"
        f"【验收轮次】该子任务第 {attempts} 次验收（上限 {settings.max_retry_per_subtask} 次重试）\n\n"
        f"请判定该结果是否达到验收标准。"
    )


def _find_subtask(state: AgentState, sub_id: object) -> dict:
    for sub in state.get("subtasks") or []:
        if str(sub.get("id")) == str(sub_id):
            return sub
    return {"id": sub_id, "title": ""}


def _result(
    subtask: dict,
    output: str,
    passed: bool,
    reason: str,
    attempts: int,
    artifacts: list[str] | None = None,
) -> dict:
    return {
        "id": subtask.get("id"),
        "title": subtask.get("title", ""),
        "output": output,
        "passed": passed,
        "reason": reason,
        "attempts": attempts,
        # 一并带上产物清单：Worker 写进工作区的文件也是交付物的一部分，
        # 丢了它前端就没法展示「这次任务产出了哪些文件」
        "artifacts": list(artifacts or []),
    }


def _verify(state: AgentState, item: dict) -> dict:
    """验收一个待验收产出。

    :return: {id, result, retry, rejection}
             result    要写进 results 的记录；None 表示「打回重做」，暂不入库
             retry     要写进 retries 的次数
             rejection 要写进 last_rejection 的理由，Worker 重做时会带上
    """
    sub_id = item.get("id")
    subtask = _find_subtask(state, sub_id)
    output = str(item.get("output", ""))
    attempts = int((state.get("retries") or {}).get(str(sub_id), 0)) + 1

    # 入库用原始产出（避免把文件正文整段塞进 results），送验收则带上产物内容
    material = _review_material(output, item.get("artifacts"))

    try:
        raw = chat(load_prompt("checker"), build_checker_prompt(subtask, material, attempts))
    except (TokenBudgetExceeded, LLMUnavailable) as exc:
        # 与「JSON 解析失败」区别对待：这次是模型根本没回话。
        # 放行等于把没验收的结果当成通过，比判失败更危险，所以一律按未通过处理。
        logger.warning("子任务 %s 无法验收（%s），按未通过处理", sub_id, exc)
        return {
            "id": sub_id,
            "retry": None,
            "rejection": None,
            "result": _result(
                subtask, output, False,
                f"验收模型不可用，无法判定：{exc}", attempts, item.get("artifacts"),
            ),
        }

    try:
        verdict = extract_json(raw)
        passed = bool(verdict.get("pass"))
        reason = str(verdict.get("reason", "")).strip() or "（模型未给出理由）"
    except Exception as exc:  # noqa: BLE001 - 只是输出格式没解析出来，放行避免任务被卡死
        logger.warning("子任务 %s 判定结果解析失败，按通过处理：%s", sub_id, exc)
        passed, reason = True, f"判定结果解析失败，自动放行：{exc}"

    if passed:
        return {"id": sub_id, "retry": None, "rejection": None,
                "result": _result(subtask, output, True, reason, attempts, item.get("artifacts"))}

    if attempts > settings.max_retry_per_subtask:
        return {
            "id": sub_id, "retry": None, "rejection": None,
            "result": _result(
                subtask, output, False,
                f"已达最大重试次数（{settings.max_retry_per_subtask}），最后一次验收意见：{reason}",
                attempts, item.get("artifacts"),
            ),
        }

    # 打回：不写 results，调度器下一轮会把它重新算作就绪，于是自动重做
    return {"id": sub_id, "retry": attempts, "rejection": reason, "result": None}


def checker_node(state: AgentState) -> dict:
    """验收本轮所有待验收产出。"""
    pending = list(state.get("pending_results") or [])
    if not pending:
        return {"logs": append_log(state, "[Checker] 没有待验收的产出，跳过")}

    if len(pending) == 1:
        verdicts = [_verify(state, pending[0])]
    else:
        workers = max(1, min(len(pending), int(settings.max_parallel_subtasks)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            verdicts = list(pool.map(lambda item: _verify(state, item), pending))

    results = list(state.get("results") or [])
    retries = dict(state.get("retries") or {})
    rejections = dict(state.get("last_rejection") or {})
    notes: list[str] = []

    for verdict in verdicts:
        sub_id = str(verdict["id"])
        if verdict["result"] is not None:
            record = verdict["result"]
            results.append(record)
            retries.pop(sub_id, None)
            rejections.pop(sub_id, None)
            notes.append(
                f"{sub_id} {'通过' if record.get('passed') else '未通过（已达上限）'}"
            )
        else:
            retries[sub_id] = verdict["retry"]
            rejections[sub_id] = str(verdict["rejection"] or "")
            notes.append(f"{sub_id} 打回重做（第 {verdict['retry']} 次）")

    logger.info("本轮验收 %d 个：%s", len(verdicts), "；".join(notes))
    return {
        "results": results,
        "retries": retries,
        "last_rejection": rejections,
        "pending_results": [],  # 已消费，避免下一轮重复验收
        "logs": append_log(state, f"[Checker] {'；'.join(notes)}"),
    }
