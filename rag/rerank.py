"""Rerank：RRF 融合之后再精排一次。

为什么 RRF 之后还要重排：
    RRF 只回答「谁在两路里都排得靠前」，它**完全没看内容**。
    对于「BM25 靠关键词命中、向量靠字面相似」这种表面命中，RRF 照样会把它顶上来。
    重排就是在小候选集上做一次更贵的相关性判断 —— 多看几眼，把真正切题的挑出来。

四档实现，精度与成本递增：

    none           不重排，直接用融合顺序（基线）
    heuristic      零依赖启发式：融合分主导 + 查询词覆盖 + 标题命中，本地微秒级
    cross_encoder  交叉编码器：query 和 doc 拼在一起过模型，精度最高，
                   需要 sentence-transformers 并下载模型（数百 MB）；
                   ⚠️ 与「clone 下来就能跑」冲突，所以只是可选项，不是默认
    llm            让现有 LLM 给候选打分，不需要额外模型，但每次检索多一次 API 调用

默认 `heuristic`：先保证零依赖可用；装了重依赖再切 `cross_encoder`，那才是精度跃迁。

**一个必须记住的坑**（campus-care 项目上实测踩过）：
    启发式重排必须让「融合分」占主导。如果让字面覆盖率主导，
    等于把向量那一路的语义贡献直接抹掉 —— Recall@1 会不升反降。
    另外融合分要按本次候选的最高分**动态归一**，用固定分母会让所有候选挤成同一个值。
"""
from __future__ import annotations

import re
from typing import Any, Protocol

from app.config import settings
from app.observability import get_logger

logger = get_logger("rag.rerank")


class Reranker(Protocol):
    """重排接口。实现方只需保证：输入候选、输出排好序的前 top_k 条。"""

    name: str

    def rerank(self, query: str, hits: list[dict], top_k: int) -> list[dict]: ...


# ---------------------------------------------------------------------------
# none：不重排
# ---------------------------------------------------------------------------
class NoopReranker:
    name = "none"

    def rerank(self, query: str, hits: list[dict], top_k: int) -> list[dict]:
        return hits[:top_k]


# ---------------------------------------------------------------------------
# heuristic：零依赖启发式
# ---------------------------------------------------------------------------
def _coverage(query_tokens: list[str], doc_tokens: list[str]) -> float:
    """查询词在文档里的覆盖率。只看「查询里的词有多少出现在文档中」。"""
    if not query_tokens:
        return 0.0
    doc_set = set(doc_tokens)
    hit = sum(1 for token in set(query_tokens) if token in doc_set)
    return hit / len(set(query_tokens))


def _heuristic_score(query: str, hits: list[dict]) -> list[dict]:
    """融合分主导 + 覆盖率微调 + 标题命中加成。"""
    from rag.embedding import tokenize

    if not hits:
        return hits

    # 动态归一：除以本次候选里的最高融合分。
    # 用固定分母（比如 1/(k+1)）会让所有候选都变成 1.0，这个维度当场失去区分度。
    max_fusion = max(float(hit.get("score", 0.0)) for hit in hits) or 1.0

    query_tokens = tokenize(query)
    query_set = set(query_tokens)
    metadata_of = lambda hit: hit.get("metadata") or {}

    for hit in hits:
        fusion_norm = float(hit.get("score", 0.0)) / max_fusion
        cover = _coverage(query_tokens, tokenize(str(hit.get("text", ""))))
        title_hit = 1.0 if query_set & set(tokenize(str(metadata_of(hit).get("title", "")))) else 0.0

        hit["coverage"] = round(cover, 4)
        hit["rerankScore"] = round(fusion_norm * 0.75 + cover * 0.15 + title_hit * 0.10, 6)

    hits.sort(key=lambda item: float(item.get("rerankScore", 0.0)), reverse=True)
    return hits


class HeuristicReranker:
    name = "heuristic"

    def rerank(self, query: str, hits: list[dict], top_k: int) -> list[dict]:
        return _heuristic_score(query, hits)[:top_k]


# ---------------------------------------------------------------------------
# cross_encoder：真正的交叉编码器
# ---------------------------------------------------------------------------
class CrossEncoderReranker:
    """交叉编码器重排：把 (query, doc) 拼在一起送进模型打分。

    与向量检索的区别：向量是「双塔」—— query 和 doc 各自编码再算余弦，
    两者从没见过面；交叉编码器让它们在同一段输入里互相 attention，精度明显更高，
    代价是不能预计算，必须对每个候选现算一次。

    模型延迟加载：没装 sentence-transformers 时不该在 import 阶段就崩，
    而是在第一次真正重排时降级（见 rerank 的 try/except）。
    """

    name = "cross_encoder"

    def __init__(self) -> None:
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("加载交叉编码器：%s", settings.rerank_model)
            self._model = CrossEncoder(settings.rerank_model)
        return self._model

    def rerank(self, query: str, hits: list[dict], top_k: int) -> list[dict]:
        if not hits:
            return hits
        try:
            model = self._load()
            scores = model.predict([(query, str(hit.get("text", ""))) for hit in hits])
        except Exception as exc:  # noqa: BLE001 - 模型缺失/下载失败都不该让检索挂掉
            logger.warning("交叉编码器不可用（%s），本次降级为启发式重排", exc)
            return _heuristic_score(query, hits)[:top_k]

        for hit, score in zip(hits, scores):
            hit["rerankScore"] = round(float(score), 6)
        hits.sort(key=lambda item: float(item.get("rerankScore", 0.0)), reverse=True)
        return hits[:top_k]


# ---------------------------------------------------------------------------
# llm：让模型给候选打分
# ---------------------------------------------------------------------------
_INDEX_RE = re.compile(r"\d+")


class LLMReranker:
    """用现有 LLM 做相关性打分。

    不引入新模型，但每次检索都要多一次 API 调用 —— 慢且花钱，
    只适合「检索次数少但精度要求高」的场景。
    """

    name = "llm"

    def rerank(self, query: str, hits: list[dict], top_k: int) -> list[dict]:
        if len(hits) <= 1:
            return hits[:top_k]

        listing = "\n\n".join(
            f"[{index}] {str(hit.get('text', ''))[:300]}" for index, hit in enumerate(hits)
        )
        prompt = (
            "你是检索结果重排器。下面是一组候选片段，请按它们与用户问题的相关程度"
            "从高到低排序，只输出片段的编号，用英文逗号分隔，不要解释。\n\n"
            f"用户问题：{query}\n\n候选片段：\n{listing}\n\n"
            "只输出形如：2,0,3 的编号序列。"
        )
        try:
            from app.llm import chat

            raw = chat("你是严格的检索结果重排器。", prompt)
            order = [int(item) for item in _INDEX_RE.findall(raw)]
        except Exception as exc:  # noqa: BLE001 - 重排失败不该让检索失败
            logger.warning("LLM 重排失败（%s），本次降级为启发式重排", exc)
            return _heuristic_score(query, hits)[:top_k]

        ranked: list[dict] = []
        used: set[int] = set()
        for index in order:
            # 模型可能返回越界编号或重复编号，这里统一过滤
            if 0 <= index < len(hits) and index not in used:
                used.add(index)
                ranked.append(hits[index])
        # 模型没提到的候选按融合顺序垫在后面，保证不丢结果
        ranked.extend(hit for index, hit in enumerate(hits) if index not in used)
        return ranked[:top_k]


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------
_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    """按配置返回重排实现（进程内缓存一份）。"""
    global _reranker
    if _reranker is not None:
        return _reranker

    provider = (settings.rerank_provider or "heuristic").strip().lower()
    mapping: dict[str, type] = {
        "none": NoopReranker,
        "heuristic": HeuristicReranker,
        "cross_encoder": CrossEncoderReranker,
        "llm": LLMReranker,
    }
    impl = mapping.get(provider)
    if impl is None:
        logger.warning("未知的 RERANK_PROVIDER=%s，回退为 heuristic", provider)
        impl = HeuristicReranker

    _reranker = impl()
    logger.info("重排实现：%s（候选池 %d）", _reranker.name, settings.rerank_pool)
    return _reranker


def reset_reranker() -> None:
    """配置变更或测试隔离时调用。"""
    global _reranker
    _reranker = None
