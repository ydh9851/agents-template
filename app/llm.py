"""LLM 接入层。

- 有 DEEPSEEK_API_KEY：走真实的 DeepSeek（OpenAI 兼容协议）。
- 没有 Key（或显式设置 MOCK_LLM=true）：自动降级为 MockLLM，
  让「Manager → Worker → Checker」最小链路在离线环境下也能完整跑通。
"""
import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import settings
from app.utils import logger

_ROLE_RE = re.compile(r"ROLE=(\w+)", re.I)


def _text_of(message: Any) -> str:
    """兼容 BaseMessage 与普通 dict。"""
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", message))


def _role_of(system_prompt: str) -> str:
    match = _ROLE_RE.search(system_prompt or "")
    return match.group(1).lower() if match else ""


def _clip(text: str, limit: int = 80) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class MockLLM:
    """离线 Mock 模型：根据 Prompt 里的 `<!--ROLE=xxx-->` 标记返回结构合法的假数据。"""

    name = "mock"

    def invoke(self, messages: list[Any]) -> AIMessage:
        system = _text_of(messages[0]) if messages else ""
        user = "\n".join(_text_of(m) for m in messages[1:])
        role = _role_of(system)

        if role == "manager":
            content = self._manager(user)
        elif role == "worker":
            content = self._worker(user)
        elif role == "checker":
            content = self._checker(user)
        elif role == "summarizer":
            content = self._summarizer(user)
        else:
            content = "（Mock 模式）已收到你的请求。配置 DEEPSEEK_API_KEY 后可获得真实回答。"
        return AIMessage(content=content)

    # ---------- 各角色的假输出 ----------
    @staticmethod
    def _checker(user: str) -> str:
        """默认放行；若配置了 MOCK_CHECKER_FAIL_TIMES，则前 N 轮故意判不通过。"""
        round_match = re.search(r"第\s*(\d+)\s*次验收", user)
        attempt = int(round_match.group(1)) if round_match else 1
        if attempt <= settings.mock_checker_fail_times:
            return json.dumps(
                {
                    "pass": False,
                    "reason": (
                        f"（Mock 故意打回）第 {attempt} 轮：结果还缺少关键依据，"
                        "请补充资料引用后重做。"
                    ),
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {"pass": True, "reason": "Mock 模式下默认判定通过：结果已覆盖验收标准。"},
            ensure_ascii=False,
        )


    @staticmethod
    def _manager(user: str) -> str:
        task = _clip(user.replace("用户任务：", ""), 60)
        subtasks = [
            {
                "id": "t1",
                "title": "信息收集与需求澄清",
                "description": f"阅读并拆解任务「{task}」，整理出完成该任务所需的关键信息与约束。",
                "deps": [],
                "acceptance": "至少列出 3 条关键信息或约束，且与任务直接相关",
                "need_rag": True,
            },
            {
                "id": "t2",
                "title": "核心方案设计",
                "description": "基于 t1 的结论，给出完成任务的可执行方案与步骤。",
                "deps": ["t1"],
                "acceptance": "给出不少于 3 个可执行步骤，并说明每步的产出",
                "need_rag": True,
            },
            {
                "id": "t3",
                "title": "结果整理与交付",
                "description": "把前两步的结果整理成结构清晰、可直接使用的最终交付物。",
                "deps": ["t1", "t2"],
                "acceptance": "输出结构清晰，包含结论与要点，可直接交付",
                "need_rag": False,
            },
        ]
        return json.dumps({"subtasks": subtasks}, ensure_ascii=False)

    @staticmethod
    def _worker(user: str) -> str:
        context = ""
        marker = "【检索资料】"
        if marker in user:
            context = user.split(marker, 1)[1].strip()
            context = _clip(context, 160)
        return (
            "结论：已根据任务描述"
            + ("与检索到的资料" if context else "")
            + "完成本子任务（Mock 输出）。\n\n"
            + ("依据：[1] " + context + "\n\n" if context else "")
            + "步骤：\n1. 明确输入与目标；\n2. 按验收标准产出结果；\n3. 自检是否遗漏。\n\n"
            + "自检：满足验收标准（Mock 模式）。"
        )

    @staticmethod
    def _summarizer(user: str) -> str:
        return (
            "## 最终交付\n\n"
            "（Mock 模式汇总）已按子任务顺序完成整理：\n\n"
            + _clip(user, 400)
            + "\n\n> 提示：配置 DEEPSEEK_API_KEY 后，这里会输出真实的汇总结果。"
        )


def get_llm() -> Any:
    """返回一个 LangChain ChatModel；无 Key 时返回 MockLLM。"""
    if settings.use_mock:
        return MockLLM()

    # 延迟导入：没装 langchain-openai 时也能启动 Mock 模式
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.model_name,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=settings.temperature,
        timeout=settings.request_timeout,
        max_retries=2,
    )


def chat(system_prompt: str, user_prompt: str) -> str:
    """一次同步对话，返回纯文本。三个 Agent 节点都通过它调用模型。"""
    llm = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    try:
        response = llm.invoke(messages)
    except Exception as exc:  # 网络/额度等异常不打断整条链路
        logger.error("调用模型失败：%s", exc)
        raise
    content = _text_of(response)
    return content
