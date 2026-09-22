"""LLM 接入层：真实 DeepSeek（OpenAI 兼容）/ 离线 Mock + 可靠性包装。

- 有 DEEPSEEK_API_KEY：走真实的 DeepSeek。
- 没有 Key（或显式 MOCK_LLM=true）：降级为 MockLLM，
  让「Manager → Worker → Checker」最小链路在离线环境下也能完整跑通。

在「能调通」之上，这层还要负责三件事：

    1. 可靠性 —— 单次超时、失败重试（指数退避）、总时间预算、多模型回退链。
       一个任务要调十几次模型，任何一次网络抖动都不该让整个任务失败；
       但也不能无脑重试把请求拖死，所以还有一条总时间预算兜底。

    2. 成本可见 —— 每次调用的 token 累计到任务上下文，随 done 事件返回；
       配置了 task_token_budget 后，超预算直接拒绝继续调用真实模型，
       宁可产出不完整的结果，也不能让一个任务烧到天价。

    3. 指标 —— 调用次数 / 耗时 / token 全部记进 metrics，/metrics 可查。
"""
from __future__ import annotations

import contextvars
import json
import re
import threading
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import settings
from app.observability import get_logger, metrics

logger = get_logger("app.llm")

_ROLE_RE = re.compile(r"ROLE=(\w+)", re.I)


class LLMUnavailable(RuntimeError):
    """所有模型、所有重试都失败。节点可以选择降级，而不是把异常抛给用户。"""


class TokenBudgetExceeded(RuntimeError):
    """当前任务的 token 预算已用尽。"""


# ---------------------------------------------------------------------------
# 任务级 token 计量
#
# 这里有个必须踩过一次才知道的坑：不能用简单的 ContextVar[int] 做累加。
# LangGraph 会把节点放进独立的 context 执行（内部走 copy_context），
# 在节点里 set 的 contextvar 不会回传到外层 —— 结果就是任务跑完 token 恒为 0，
# 「成本可见」和「预算熔断」两个功能一起静默失效。
#
# 解法是往 contextvar 里放一个**可变对象**：各节点改的是同一个对象的字段，
# 靠引用共享，写入自然对父 context 可见。（读操作用 contextvar 没问题，
# 出问题的只有写。）
# ---------------------------------------------------------------------------
class _TokenUsage:
    """线程安全的 token 累加器。"""

    __slots__ = ("total", "_lock")

    def __init__(self) -> None:
        self.total = 0
        self._lock = threading.Lock()

    def add(self, count: int) -> None:
        if count <= 0:
            return
        with self._lock:
            self.total += count


_usage: contextvars.ContextVar[_TokenUsage | None] = contextvars.ContextVar(
    "task_usage", default=None
)


def _current_usage() -> _TokenUsage:
    usage = _usage.get()
    if usage is None:
        usage = _TokenUsage()
        _usage.set(usage)
    return usage


def reset_token_usage() -> None:
    """每次任务开始前换一个新累加器，避免把上一个任务的用量算进来。"""
    _usage.set(_TokenUsage())


def get_token_usage() -> int:
    return _current_usage().total


def _add_tokens(count: int) -> None:
    _current_usage().add(count)


def _ensure_budget() -> None:
    budget = int(settings.task_token_budget)
    if budget <= 0:  # 0 = 不限制
        return
    used = get_token_usage()
    if used >= budget:
        raise TokenBudgetExceeded(f"任务 token 预算已用尽（已用 {used}，上限 {budget}）")


# ---------------------------------------------------------------------------
# 消息辅助
# ---------------------------------------------------------------------------
def _text_of(message: Any) -> str:
    """兼容 BaseMessage 与普通 dict。"""
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", message))


def _tokens_of(response: Any) -> int:
    """从 AIMessage 里取 token 用量。

    优先 usage_metadata（langchain-core 0.3 的标准字段），
    退回 response_metadata.token_usage（各家 SDK 的原始返回）。
    两者都没有就返回 0 —— 成本统计宁可为 0，也不要瞎估一个数出来。
    """
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        total = usage.get("total_tokens")
        if total:
            return int(total)
    metadata = getattr(response, "response_metadata", None) or {}
    token_usage = metadata.get("token_usage") or {}
    return int(token_usage.get("total_tokens") or 0)


def _role_of(system_prompt: str) -> str:
    match = _ROLE_RE.search(system_prompt or "")
    return match.group(1).lower() if match else ""


def _clip(text: str, limit: int = 80) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# 离线 Mock
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 真实模型客户端（按模型名缓存，避免每次调用都重建连接池）
# ---------------------------------------------------------------------------
_clients: dict[str, Any] = {}


def _client_for(model: str) -> Any:
    if model in _clients:
        return _clients[model]

    # 延迟导入：没装 langchain-openai 时也能以 Mock 模式启动
    from langchain_openai import ChatOpenAI

    client = ChatOpenAI(
        model=model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=settings.temperature,
        timeout=settings.request_timeout,
        # 重试由本模块统一负责：关掉 SDK 自带的，
        # 否则「SDK 重试 × 我的重试 × 模型数」会叠乘成完全不可控的等待时间
        max_retries=0,
    )
    _clients[model] = client
    return client


def reset_clients() -> None:
    """清掉客户端缓存（测试或切换配置时用）。"""
    _clients.clear()


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------
def chat(system_prompt: str, user_prompt: str) -> str:
    """一次同步对话，返回纯文本。三个 Agent 节点都通过它调用模型。

    真实模式下会依次尝试「模型回退链 × 重试次数」，全程受总时间预算约束。
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    if settings.use_mock:
        text = _text_of(MockLLM().invoke(messages))
        metrics.inc("llm_calls_total", model="mock", status="ok")
        return text

    _ensure_budget()

    started = time.perf_counter()
    budget = float(settings.llm_total_timeout)
    chain = settings.fallback_model_list
    last_error: Exception | None = None

    for attempt in range(max(0, int(settings.llm_max_retries)) + 1):
        if time.perf_counter() - started >= budget:
            logger.warning("已达 LLM 总时间预算 %.0fs，停止重试", budget)
            break

        for model in chain:
            call_started = time.perf_counter()
            try:
                response = _client_for(model).invoke(messages)
            except Exception as exc:  # noqa: BLE001 - 网络/额度/限流都在这兜住
                elapsed = time.perf_counter() - call_started
                last_error = exc
                metrics.inc("llm_calls_total", model=model, status="error")
                metrics.observe("llm_latency_seconds", elapsed, model=model)
                logger.warning("模型 %s 调用失败（第 %d 次尝试）：%s", model, attempt + 1, exc)
                continue

            elapsed = time.perf_counter() - call_started
            text = _text_of(response)
            tokens = _tokens_of(response)
            _add_tokens(tokens)
            metrics.inc("llm_calls_total", model=model, status="ok")
            metrics.observe("llm_latency_seconds", elapsed, model=model)
            metrics.inc("llm_tokens_total", tokens, model=model)

            if attempt > 0 or model != chain[0]:
                logger.info(
                    "LLM 调用成功（第 %d 次尝试，模型 %s，耗时 %.2fs，tokens=%d）",
                    attempt + 1, model, elapsed, tokens,
                )
            return text

        if attempt < settings.llm_max_retries:
            wait = max(0.0, float(settings.llm_retry_backoff)) * (2 ** attempt)
            if time.perf_counter() - started + wait >= budget:
                logger.warning("退避等待会超出总时间预算，停止重试")
                break
            time.sleep(wait)

    raise LLMUnavailable(
        f"所有模型均调用失败（已尝试 {len(chain)} 个模型，耗时 "
        f"{time.perf_counter() - started:.1f}s）：{last_error}"
    )


def chat_with_tools(messages: list[Any], tools: list[dict] | None = None) -> Any:
    """带工具绑定的单次调用，返回完整 AIMessage（可能含 `tool_calls`）。

    为什么单独开一个入口而不是扩展 chat()：
        chat() 的契约是「问答返回一段纯文本」，而工具调用需要拿到
        `AIMessage.tool_calls` 这个结构化字段，两者的返回类型根本不同。
        强行合并会让所有调用方都要判断「这次返回的是文本还是消息对象」。

    这里也**不做重试**：工具循环本身有轮次上限，
    某一轮失败时让上层的循环决定是继续还是收尾，比在底层盲目重试更合理。

    :param tools: 传空列表表示「不绑定工具」，等价于普通的一次调用 ——
                  工具循环在最后一轮会这么用（让它基于已有信息收尾）。
    """
    _ensure_budget()

    model = settings.model_name
    client = _client_for(model)
    if tools:
        client = client.bind_tools(tools)

    started = time.perf_counter()
    try:
        response = client.invoke(messages)
    except Exception as exc:  # noqa: BLE001
        metrics.inc("llm_calls_total", model=model, status="error")
        metrics.observe("llm_latency_seconds", time.perf_counter() - started, model=model)
        raise LLMUnavailable(f"带工具的模型调用失败（{model}）：{exc}") from exc

    elapsed = time.perf_counter() - started
    tokens = _tokens_of(response)
    _add_tokens(tokens)
    metrics.inc("llm_calls_total", model=model, status="ok")
    metrics.observe("llm_latency_seconds", elapsed, model=model)
    metrics.inc("llm_tokens_total", tokens, model=model)
    return response


def content_of(response: Any) -> str:
    """从模型返回里取出纯文本。兼容「content 是字符串」与「content 是分段列表」两种形态。"""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


def get_llm() -> Any:
    """返回一个 LangChain ChatModel；无 Key 时返回 MockLLM。

    只在需要直接操作 ChatModel 的场景使用（自定义 callback、流式等）；
    节点统一走 chat()，只有那里带重试、预算与指标。
    """
    if settings.use_mock:
        return MockLLM()
    return _client_for(settings.model_name)
