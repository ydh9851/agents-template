"""Embedding 抽象层。

- local（默认）：本地哈希向量，零额外依赖、不需要联网，中文用「单字 + 相邻双字」切词；
- openai：调用任意 OpenAI 兼容的 /embeddings 接口（DeepSeek 官方暂无 embedding，
  需要把 EMBEDDING_BASE_URL / EMBEDDING_API_KEY 指向支持的服务）。
"""
import hashlib
import re

import numpy as np

from app.config import settings
from app.utils import get_logger

logger = get_logger("rag.embedding")

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")
_CN_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """中英文混合分词：英文按词，中文按「单字 + 相邻双字」。"""
    text = (text or "").lower()
    tokens: list[str] = _WORD_RE.findall(text)
    for run in _CN_RUN_RE.findall(text):
        tokens.extend(run)  # 单字
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))  # 相邻双字
    return tokens


class LocalHashEmbedding:
    """符号哈希（hashing trick）向量：确定性、可离线、对中文关键词敏感。

    它不是语义向量，但在 BM25 之外能补足「字面不重合但词序相近」的召回，
    并且让整套 RAG 不依赖 2GB 的模型下载。
    """

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or settings.embedding_dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        return vector.tolist()


class OpenAICompatEmbedding:
    """调用 OpenAI 兼容的 embedding 接口。"""

    def __init__(self) -> None:
        from openai import OpenAI

        api_key = settings.embedding_api_key or settings.deepseek_api_key
        base_url = settings.embedding_base_url or settings.deepseek_base_url
        if not api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=openai 时必须配置 EMBEDDING_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=settings.request_timeout)
        self.model = settings.embedding_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        batch_size = max(1, settings.embedding_batch_size)
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=[text])
        return response.data[0].embedding


_embedder = None


def get_embedder():
    """返回全局唯一的 embedder（失败时回退到本地哈希向量）。"""
    global _embedder
    if _embedder is not None:
        return _embedder

    if settings.embedding_provider.lower() == "openai":
        try:
            _embedder = OpenAICompatEmbedding()
            logger.info("使用 OpenAI 兼容 embedding：%s", settings.embedding_model)
            return _embedder
        except Exception as exc:
            logger.warning("初始化远程 embedding 失败（%s），回退到本地哈希向量", exc)

    _embedder = LocalHashEmbedding()
    logger.info("使用本地哈希向量（dim=%d）", _embedder.dim)
    return _embedder


def reset_embedder() -> None:
    """测试用：清掉缓存。"""
    global _embedder
    _embedder = None
