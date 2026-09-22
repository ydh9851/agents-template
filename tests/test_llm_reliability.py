"""LLM 可靠性测试：重试、模型回退、token 计量、预算熔断。

全部用假客户端注入，不联网、不消耗额度。
这些能力平时「看起来正常」——只有在网络抖动、额度耗尽、任务跑飞时才暴露，
所以必须用确定性测试把行为钉死。
"""
import pytest

from app import llm as llm_module
from app.config import settings
from app.llm import LLMUnavailable, TokenBudgetExceeded, chat, get_token_usage, reset_token_usage


class _FakeResponse:
    def __init__(self, content: str, tokens: int) -> None:
        self.content = content
        self.usage_metadata = {"total_tokens": tokens}


class _FakeClient:
    """按「第几次调用」决定失败还是成功，用来构造确定性的重试场景。"""

    def __init__(self, fail_times: int = 0, tokens: int = 10) -> None:
        self.fail_times = fail_times
        self.tokens = tokens
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("模拟网络故障")
        return _FakeResponse("ok", self.tokens)


@pytest.fixture
def real_mode(monkeypatch):
    """让 chat() 走「真实模式」分支，但底层客户端是假的。"""
    monkeypatch.setattr(settings, "mock_llm", False)
    monkeypatch.setattr(settings, "deepseek_api_key", "sk-test")
    monkeypatch.setattr(settings, "llm_max_retries", 2)
    monkeypatch.setattr(settings, "llm_retry_backoff", 0.0)  # 测试里不等退避
    monkeypatch.setattr(settings, "task_token_budget", 0)
    monkeypatch.setattr(settings, "fallback_models", "")
    reset_token_usage()
    yield
    llm_module.reset_clients()


def _patch(monkeypatch, client) -> None:
    monkeypatch.setattr(llm_module, "_client_for", lambda model: client)


def test_retry_succeeds_after_transient_failure(real_mode, monkeypatch):
    client = _FakeClient(fail_times=2)
    _patch(monkeypatch, client)
    assert chat("system", "user") == "ok"
    assert client.calls == 3  # 首次失败 + 第 1 次重试失败 + 第 2 次重试成功


def test_all_retries_failed_raises(real_mode, monkeypatch):
    client = _FakeClient(fail_times=99)
    _patch(monkeypatch, client)
    with pytest.raises(LLMUnavailable):
        chat("system", "user")
    assert client.calls == 3  # 首次 + max_retries(2)


def test_fallback_model_is_used(real_mode, monkeypatch):
    monkeypatch.setattr(settings, "fallback_models", "backup-model")
    primary = _FakeClient(fail_times=99)
    backup = _FakeClient()

    def fake_client_for(model):
        return primary if model == settings.model_name else backup

    monkeypatch.setattr(llm_module, "_client_for", fake_client_for)
    assert chat("system", "user") == "ok"
    # 主模型失败一次后立刻切备用模型，不用等一轮退避重试
    assert primary.calls == 1
    assert backup.calls == 1


def test_tokens_accumulate_across_calls(real_mode, monkeypatch):
    _patch(monkeypatch, _FakeClient(tokens=42))
    chat("system", "user")
    chat("system", "user")
    assert get_token_usage() == 84


def test_token_budget_blocks_further_calls(real_mode, monkeypatch):
    monkeypatch.setattr(settings, "task_token_budget", 50)
    _patch(monkeypatch, _FakeClient(tokens=42))

    chat("system", "user")   # 用掉 42
    chat("system", "user")   # 累计 84，已超预算
    with pytest.raises(TokenBudgetExceeded):
        chat("system", "user")


def test_reset_token_usage_clears_counter(real_mode, monkeypatch):
    _patch(monkeypatch, _FakeClient(tokens=10))
    chat("system", "user")
    assert get_token_usage() == 10
    reset_token_usage()
    assert get_token_usage() == 0


def test_mock_mode_never_touches_client(monkeypatch):
    monkeypatch.setattr(settings, "mock_llm", True)
    client = _FakeClient()
    _patch(monkeypatch, client)  # 不应被调用

    text = chat("<!--ROLE=manager-->", "用户任务：测试拆解")
    assert "subtasks" in text
    assert client.calls == 0
