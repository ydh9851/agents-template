"""带工具的 ReAct 循环。

Worker（产出文档/方案）与 Coder（产出可运行代码）走的是同一套循环：
    模型选工具 → 执行 → 结果回灌 → 直到它给出最终答复。

抽出来共用而不是各写一份，是因为这里有三个很容易写错的地方 ——
轮次上限、工具错误回灌、用尽上限后摘掉工具收尾 ——
复制一份就等于多一份把它们写错的机会。
"""
from __future__ import annotations

from typing import Any

from app.config import settings
from app.llm import chat_with_tools, content_of
from app.tools import run_tool, tool_result_text, tool_schemas, tools_prompt_block
from app.utils import get_logger

logger = get_logger("agent.loop")


def run_with_tools(
    system_prompt: str,
    user_prompt: str,
    tag: str = "task",
    max_rounds: int | None = None,
) -> tuple[str, list[str]]:
    """跑一轮 ReAct 循环。

    :param tag:        日志里标识这是谁在跑（子任务 id / 角色名），便于排查
    :param max_rounds: 轮次上限，默认取通用的 MAX_TOOL_ROUNDS
    :return: (最终产出文本, 产物文件列表)

    轮次用尽时**不判失败**，而是摘掉工具再问一次，让它基于已有信息收尾 ——
    至少能拿到一份结论，而不是白跑一轮 token。
    """
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    system = system_prompt
    prompt_block = tools_prompt_block()
    if prompt_block:
        system = f"{system_prompt}\n\n{prompt_block}"

    messages: list[Any] = [
        SystemMessage(content=system),
        HumanMessage(content=user_prompt),
    ]
    artifacts: list[str] = []
    rounds = max(1, int(max_rounds or settings.max_tool_rounds))

    for round_index in range(rounds):
        response = chat_with_tools(messages, tool_schemas())
        calls = list(getattr(response, "tool_calls", None) or [])
        if not calls:
            # 没有工具调用 = 它认为活儿干完了，这就是最终产出
            return content_of(response), artifacts

        messages.append(response)
        for call in calls:
            name = str(call.get("name") or "")
            result = run_tool(name, call.get("args") or {})
            if result.get("artifact"):
                artifacts.append(str(result["artifact"]))
            logger.info(
                "%s 第 %d/%d 轮调用 %s → %s",
                tag, round_index + 1, rounds, name,
                "ok" if result.get("ok") else "失败",
            )
            messages.append(
                ToolMessage(
                    content=tool_result_text(result),
                    tool_call_id=str(call.get("id") or ""),
                )
            )

    logger.warning("%s 工具调用达上限 %d 轮，摘掉工具让它收尾", tag, rounds)
    messages.append(
        HumanMessage(content="工具调用轮次已达上限，请直接基于已有信息给出最终结论。")
    )
    return content_of(chat_with_tools(messages, [])), artifacts
