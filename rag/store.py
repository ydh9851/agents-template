"""向量库封装。

优先使用 ChromaDB；如果环境里没装 chromadb，自动降级为纯 JSON 的本地向量库，
保证 RAG 在任何环境下都能工作（只是数据量大时性能差一些）。
"""
import json
from pathlib import Path
from typing import Any

import numpy as np

from app.config import settings
from app.utils import get_logger

logger = get_logger("rag.store")

COLLECTION_NAME = "agents_docs"


class BaseVectorStore:
    """向量库统一接口。"""

    backend = "base"

    def add(self, ids: list[str], documents: list[str], metadatas: list[dict], embeddings: list[list[float]]) -> None:
        raise NotImplementedError

    def query(self, query_embedding: list[float], top_k: int) -> list[dict]:
        raise NotImplementedError

    def get_all(self) -> list[dict]:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB 持久化实现。"""

    backend = "chromadb"

    def __init__(self, persist_dir: str | None = None) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.client = chromadb.PersistentClient(
            path=persist_dir or settings.chroma_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids, documents, metadatas, embeddings) -> None:
        if not ids:
            return
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def query(self, query_embedding: list[float], top_k: int) -> list[dict]:
        if self.count() == 0:
            return []
        result = self.collection.query(query_embeddings=[query_embedding], n_results=min(top_k, self.count()))
        hits: list[dict] = []
        for doc_id, text, meta, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            hits.append(
                {
                    "id": doc_id,
                    "text": text,
                    "metadata": dict(meta or {}),
                    # cosine 距离转相似度，便于统一排序口径
                    "score": 1.0 - float(distance),
                }
            )
        return hits

    def get_all(self) -> list[dict]:
        if self.count() == 0:
            return []
        result = self.collection.get(include=["documents", "metadatas"])
        return [
            {"id": doc_id, "text": text, "metadata": dict(meta or {})}
            for doc_id, text, meta in zip(result["ids"], result["documents"], result["metadatas"])
        ]

    def count(self) -> int:
        return int(self.collection.count())

    def reset(self) -> None:
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )


class SimpleVectorStore(BaseVectorStore):
    """零依赖降级实现：文档 + 向量存成一个 JSON 文件。"""

    backend = "simple-json"

    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path or settings.index_dir) / "vectors.json"
        self.rows: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self.rows = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("向量文件损坏，已忽略：%s", self.path)
                self.rows = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.rows, ensure_ascii=False), encoding="utf-8")

    def add(self, ids, documents, metadatas, embeddings) -> None:
        existing = {row["id"] for row in self.rows}
        for doc_id, text, meta, vector in zip(ids, documents, metadatas, embeddings):
            if doc_id in existing:
                continue
            self.rows.append({"id": doc_id, "text": text, "metadata": meta, "vector": vector})
        self._save()

    def query(self, query_embedding: list[float], top_k: int) -> list[dict]:
        if not self.rows:
            return []
        matrix = np.asarray([row["vector"] for row in self.rows], dtype=np.float32)
        query_vec = np.asarray(query_embedding, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(query_vec) or 1.0)
        norms[norms == 0] = 1.0
        similarities = (matrix @ query_vec) / norms
        order = np.argsort(-similarities)[:top_k]
        return [
            {
                "id": self.rows[i]["id"],
                "text": self.rows[i]["text"],
                "metadata": self.rows[i].get("metadata", {}),
                "score": float(similarities[i]),
            }
            for i in order
        ]

    def get_all(self) -> list[dict]:
        return [{"id": r["id"], "text": r["text"], "metadata": r.get("metadata", {})} for r in self.rows]

    def count(self) -> int:
        return len(self.rows)

    def reset(self) -> None:
        self.rows = []
        self._save()


_store: BaseVectorStore | None = None


def get_vector_store() -> BaseVectorStore:
    """返回全局唯一的向量库实例，Chroma 不可用时自动降级。"""
    global _store
    if _store is not None:
        return _store
    try:
        _store = ChromaVectorStore()
        logger.info("向量库后端：ChromaDB（%s）", settings.chroma_dir)
    except Exception as exc:
        logger.warning("ChromaDB 不可用（%s），回退到本地 JSON 向量库", exc)
        _store = SimpleVectorStore()
    return _store


def reset_vector_store() -> None:
    """测试用：清掉缓存实例。"""
    global _store
    _store = None
