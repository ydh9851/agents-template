"""Tester 角色：在不知道实现的前提下，先把测试写出来。

为什么必须独立于 Coder：

    让写实现的人自己出题，结果大概率是「题刚好被他答对」——
    边界条件、非法输入这些真正值钱的地方，会被有意无意地绕开。
    所以 Tester 只看子任务的**验收标准**，看不到任何实现代码。

产出落在约定的 tests/ 目录，交给沙箱里的 pytest 执行。
"""
from __future__ import annotations

from agents.loop import run_with_tools
from app.config import settings
from app.utils import get_logger

logger = get_logger("agent.tester")


def write_tests(subtask: dict, context: str) -> tuple[str, list[str]]:
    """为子任务编写测试。

    :return: (说明文本, 写出的测试文件列表)

    即使执行类工具被关掉（CODE_EXECUTION_ENABLED=false），这一步依然能跑 ——
    写测试文件属于文件写入，不涉及执行代码。
    """
    prompt = (
        f"【子任务】\n"
        f"标题：{subtask.get('title', '')}\n"
        f"描述：{subtask.get('description', '')}\n"
        f"验收标准：{subtask.get('acceptance', '')}\n\n"
        f"【可用资料】\n{context or '（无）'}\n\n"
        f"请编写测试文件（保存到 tests/ 目录），完成后用一句话说明覆盖了哪些用例。"
    )

    from app.prompts import load_prompt

    text, artifacts = run_with_tools(
        load_prompt("tester"),
        prompt,
        tag=f"tester/{subtask.get('id')}",
        max_rounds=settings.testing_max_rounds,
    )

    test_files = [path for path in artifacts if path.endswith(".py")]
    logger.info(
        "Tester 为子任务 %s 产出 %d 个测试文件：%s",
        subtask.get("id"), len(test_files), test_files or "（无）",
    )
    return text, test_files
