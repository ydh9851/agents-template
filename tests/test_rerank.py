"""重排层测试。

重排是「可插拔 + 必须能降级」的典型：任何一档实现出问题（依赖没装、模型连不上、
额度耗尽），都不该让整个检索失败 —— 所以降级路径都有用例钉住。
"""
import pytest

from app.config import settings
from rag.rerank import (
    CrossEncoderReranker,
    HeuristicReranker,
    LLMReranker,
    NoopReranker,
    get_reranker,
    reset_reranker,
)


@pytest.fixture(autouse=True)
def _clean_reranker():
    reset_reranker()
    yield
    reset_reranker()


def _hits() -> list[dict]:
    return [
        {"id": "a", "text": "完全无关的内容", "metadata": {"title": "别的"}, "score": 0.030},
        {"id": "b", "text": "RRF 倒数排名融合的具体做法", "metadata": {"title": "RRF 融合"}, "score": 0.028},
        {"id": "c", "text": "另一种东西", "metadata": {"title": "无关"}, "score": 0.010},
    ]


def test_noop_keeps_fusion_order():
    result = NoopReranker().rerank("RRF 融合", _hits(), 2)
    assert [hit["id"] for hit in result] == ["a", "b"]


def test_heuristic_promotes_more_relevant_hit():
    """两条候选融合分接近时，字面更贴题、标题命中的应该被提上来。"""
    result = HeuristicReranker().rerank("RRF 倒数排名融合", _hits(), 3)
    assert result[0]["id"] == "b"


def test_heuristic_annotates_scores():
    result = HeuristicReranker().rerank("RRF 融合", _hits(), 3)
    assert all("rerankScore" in hit and "coverage" in hit for hit in result)


def test_heuristic_handles_empty_input():
    assert HeuristicReranker().rerank("随便", [], 3) == []


def test_heuristic_respects_top_k():
    assert len(HeuristicReranker().rerank("RRF", _hits(), 2)) == 2


def test_noop_and_heuristic_do_not_mutate_length():
    hits = _hits()
    NoopReranker().rerank("q", hits, 3)
    assert len(hits) == 3


def test_cross_encoder_degrades_without_dependency():
    """没装 sentence-transformers 时必须降级成启发式，而不是抛异常。"""
    result = CrossEncoderReranker().rerank("RRF 倒数排名融合", _hits(), 2)
    assert len(result) == 2
    assert result[0]["id"] == "b"  # 降级后走启发式逻辑


def test_llm_reranker_degrades_on_failure(monkeypatch):
    """LLM 重排失败（额度耗尽 / 网络异常）时也要降级，不能让检索整体挂掉。"""

    def boom(*args, **kwargs):
        raise RuntimeError("模拟模型不可用")

    monkeypatch.setattr("app.llm.chat", boom)
    result = LLMReranker().rerank("RRF 倒数排名融合", _hits(), 2)
    assert len(result) == 2


def test_llm_reranker_ignores_invalid_indexes(monkeypatch):
    """模型返回越界/重复编号时不能丢结果，也不能崩。"""
    monkeypatch.setattr("app.llm.chat", lambda *a, **k: "9,0,0,-1")
    result = LLMReranker().rerank("q", _hits(), 3)
    assert len(result) == 3
    assert result[0]["id"] == "a"  # 只认合法编号 0


def test_factory_respects_config(monkeypatch):
    monkeypatch.setattr(settings, "rerank_provider", "none")
    reset_reranker()
    assert get_reranker().name == "none"


def test_factory_falls_back_on_unknown_provider(monkeypatch):
    monkeypatch.setattr(settings, "rerank_provider", "不存在的后端")
    reset_reranker()
    assert get_reranker().name == "heuristic"


def test_factory_caches_instance(monkeypatch):
    monkeypatch.setattr(settings, "rerank_provider", "heuristic")
    reset_reranker()
    assert get_reranker() is get_reranker()
