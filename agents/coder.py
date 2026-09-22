"""Coder 角色：写实现，并跑到测试通过为止。

与 Worker 的本质区别在**验收依据**：

    Worker 产出的是「一段话」，靠 Checker 读文字判断合格与否 —— 那是软的；
    Coder 产出的是「能跑起来的东西」，靠 pytest 的真实退出码判断 —— 那是硬的。

模型没法用漂亮话把红色测试说成绿色。这就是这个角色存在的全部意义，
也是这套框架里第一次出现「不由 LLM 自己说了算」的验收环节。
"""
from __future__ import annotations

from agents.loop import run_with_tools
from app.config import settings
from app.utils import get_logger

logger = get_logger("agent.coder")


def implement(subtask: dict, context: str, test_summary: str) -> tuple[str, list[str]]:
    """写实现并反复验证，直到测试通过（或轮次用尽）。

    :param test_summary: Tester 写的测试说明，让 Coder 知道要满足什么
    :return: (说明文本, 产物文件列表)
    """
    prompt = (
        f"【子任务】\n"
        f"标题：{subtask.get('title', '')}\n"
        f"描述：{subtask.get('description', '')}\n"
        f"验收标准：{subtask.get('acceptance', '')}\n\n"
        f"【Tester 写的测试】\n{test_summary or '（Tester 没有产出测试，请自行判断边界情况）'}\n\n"
        f"【可用资料】\n{context or '（无）'}\n\n"
        f"请实现该子任务，并**务必**调用 run_tests 验证到全部通过。"
    )

    from app.prompts import load_prompt

    text, artifacts = run_with_tools(
        load_prompt("coder"),
        prompt,
        tag=f"coder/{subtask.get('id')}",
        max_rounds=settings.coding_max_rounds,
    )

    code_files = [path for path in artifacts if path.endswith(".py")]
    logger.info(
        "Coder 为子任务 %s 产出 %d 个代码文件：%s",
        subtask.get("id"), len(code_files), code_files or "（无）",
    )
    return text, artifacts
