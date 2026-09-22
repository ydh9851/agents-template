"""BM25 + 向量混合检索，用 RRF（Reciprocal Rank Fusion）融合两条结果。

RRF 公式：score(d) = Σ 1 / (k + rank_i(d))
其中 rank_i(d) 是文档 d 在第 i 路检索结果里的名次（从 1 开始），k 默认 60。
它只依赖名次、不依赖分数量纲，所以非常适合融合 BM25 与余弦相似度这类不同尺度的结果。
"""
from typing import Any

from app.config import settings
from app.utils import get_logger
from rag.embedding import get_embedder, tokenize
from rag.rerank import get_reranker
from rag.store import get_vector_store

logger = get_logger("rag.hybrid")


class HybridRetriever:
    """两路召回（BM25 / 向量）+ RRF 融合。"""

    def __init__(self) -> None:
        self.embedder = get_embedder()
        self.store = get_vector_store()
        self._bm25: Any = None
        self._bm25_ids: list[str] = []
        self._bm25_texts: dict[str, str] = {}
        self._bm25_metas: dict[str, dict] = {}

    # ---------- 建索引 ----------
    def rebuild_bm25(self) -> None:
        """从向量库读取全部文档，重建 BM25 倒排（保证两条检索线语料一致）。"""
        rows = self.store.get_all()
        self._bm25_ids = [row["id"] for row in rows]
        self._bm25_texts = {row["id"]: row["text"] for row in rows}
        self._bm25_metas = {row["id"]: row.get("metadata", {}) for row in rows}
        if not rows:
            self._bm25 = None
            return
        try:
            from rank_bm25 import BM25Okapi

            corpus = [tokenize(row["text"]) or [""] for row in rows]
            self._bm25 = BM25Okapi(corpus)
        except ImportError:
            logger.warning("未安装 rank-bm25，关键词检索将退化为子串匹配")
            self._bm25 = None

    def index_chunks(self, chunks: list[dict]) -> int:
        """把切分好的 chunk 写入向量库，并重建 BM25。"""
        if not chunks:
            return 0
        texts = [chunk["text"] for chunk in chunks]
        vectors = self.embedder.embed_documents(texts)
        self.store.add(
            ids=[chunk["id"] for chunk in chunks],
            documents=texts,
            metadatas=[chunk.get("metadata", {}) for chunk in chunks],
            embeddings=vectors,
        )
        self.rebuild_bm25()
        return len(chunks)

    # ---------- 检索 ----------
    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        if not self._bm25_ids:
            return []
        tokens = tokenize(query) or [query]
        if self._bm25 is not None:
            scores = self._bm25.get_scores(tokens)
            order = sorted(range(len(scores)), key=lambda i: -float(scores[i]))[:top_k]
            return [
                {
                    "id": self._bm25_ids[i],
                    "text": self._bm25_texts[self._bm25_ids[i]],
                    "metadata": self._bm25_metas.get(self._bm25_ids[i], {}),
                    "score": float(scores[i]),
                }
                for i in order
            ]
        # 降级：按关键词命中次数排序
        hits = []
        for doc_id in self._bm25_ids:
            text = self._bm25_texts[doc_id]
            hits.append(
                {
                    "id": doc_id,
                    "text": text,
                    "metadata": self._bm25_metas.get(doc_id, {}),
                    "score": float(sum(text.count(tok) for tok in tokens)),
                }
            )
        hits.sort(key=lambda item: -item["score"])
        return [hit for hit in hits[:top_k] if hit["score"] > 0]

    def _vector_search(self, query: str, top_k: int) -> list[dict]:
        if self.store.count() == 0:
            return []
        return self.store.query(self.embedder.embed_query(query), top_k)

    @staticmethod
    def rrf_fuse(ranked_lists: list[list[dict]], k: int = 60) -> list[tuple[str, float]]:
        """RRF 融合：输入多路有序结果，输出 (doc_id, 融合分) 降序列表。"""
        scores: dict[str, float] = {}
        for ranked in ranked_lists:
            for rank, item in enumerate(ranked, start=1):
                scores[item["id"]] = scores.get(item["id"], 0.0) + 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda pair: -pair[1])

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """混合检索入口：两路召回 → RRF 融合 → 重排 → 取 top_k。"""
        top_k = top_k or settings.rag_top_k
        query = (query or "").strip()
        if not query:
            return []

        # 两路各多召回一些，给 RRF 更大的融合空间
        candidate_k = max(top_k * 3, 10)
        bm25_hits = self._bm25_search(query, candidate_k)
        vector_hits = self._vector_search(query, candidate_k)

        pool: dict[str, dict] = {}
        for hit in bm25_hits + vector_hits:
            pool.setdefault(hit["id"], hit)

        fused = self.rrf_fuse([bm25_hits, vector_hits], k=settings.rrf_k)

        # 先取一个比 top_k 更大的池，再交给重排层收敛到 top_k ——
        # 重排的全部价值在于「在更大的候选集里重新判断」，池子只有 top_k 就没得排了
        pool_size = max(top_k, int(settings.rerank_pool))
        candidates: list[dict] = []
        for doc_id, score in fused[:pool_size]:
            item = pool.get(doc_id)
            if not item:
                continue
            candidates.append(
                {
                    "id": doc_id,
                    "text": item["text"],
                    "metadata": item.get("metadata", {}),
                    "source": (item.get("metadata") or {}).get("source", ""),
                    "score": round(score, 6),
                }
            )

        return get_reranker().rerank(query, candidates, top_k)

    def stats(self) -> dict:
        return {
            "backend": self.store.backend,
            "documents": self.store.count(),
            "bm25_ready": self._bm25 is not None,
        }


_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    """返回全局唯一的检索器（首次调用时自动加载索引）。"""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
        _retriever.rebuild_bm25()
    return _retriever


def reset_retriever() -> None:
    """重建索引后调用，强制下次重新加载。"""
    global _retriever
    _retriever = None
