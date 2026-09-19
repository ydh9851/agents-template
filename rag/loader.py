"""文档加载与切分。

- 从 data/docs 读取 markdown 文档；
- 用 LangChain 的 RecursiveCharacterTextSplitter 切分（chunk_size=500）；
- 如果没装 langchain-text-splitters，则退化为内置的按段落/句子切分实现。
"""
import re
from pathlib import Path

from app.config import settings
from app.utils import get_logger

logger = get_logger("rag.loader")

# 分隔符优先级：段落 -> 换行 -> 中文句号 / 分号 / 逗号 -> 英文句号 -> 空格
_SEPARATORS = ["\n\n", "\n", "。", "；", "，", ". ", " ", ""]


def _split_by_separators(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """内置的轻量切分实现（无第三方依赖），行为对齐 RecursiveCharacterTextSplitter。"""
    segments: list[str] = [text]
    for sep in _SEPARATORS:
        next_segments: list[str] = []
        for seg in segments:
            if len(seg) <= chunk_size or not sep:
                next_segments.append(seg)
                continue
            parts = seg.split(sep)
            for i, part in enumerate(parts):
                piece = part + (sep if i < len(parts) - 1 else "")
                if piece:
                    next_segments.append(piece)
        segments = next_segments

    # 合并过短的片段，并在相邻 chunk 之间保留 overlap
    chunks: list[str] = []
    buffer = ""
    for seg in segments:
        if len(buffer) + len(seg) <= chunk_size:
            buffer += seg
            continue
        if buffer.strip():
            chunks.append(buffer.strip())
        buffer = (buffer[-chunk_overlap:] if chunk_overlap > 0 else "") + seg
    if buffer.strip():
        chunks.append(buffer.strip())

    # 兜底：仍有超长片段就硬切
    result: list[str] = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            result.append(chunk)
            continue
        step = max(1, chunk_size - chunk_overlap)
        result.extend(chunk[i : i + chunk_size] for i in range(0, len(chunk), step))
    return [c for c in result if c.strip()]


def split_text(text: str, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[str]:
    """把长文本切成若干片段。"""
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=_SEPARATORS,
            length_function=len,
        )
        return [c.strip() for c in splitter.split_text(text) if c.strip()]
    except ImportError:
        logger.warning("未安装 langchain-text-splitters，使用内置切分器")
        return _split_by_separators(text, chunk_size, chunk_overlap)


def load_documents(docs_dir: str | None = None) -> list[dict]:
    """读取 docs 目录下的所有 markdown/txt 文件。"""
    root = Path(docs_dir or settings.docs_dir)
    if not root.exists():
        logger.warning("文档目录不存在：%s", root)
        return []

    documents: list[dict] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".md", ".markdown", ".txt"} or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        # 第一行如果是 # 标题，就当作文档标题
        title_match = re.match(r"^#\s+(.+)$", text.splitlines()[0]) if text else None
        documents.append(
            {
                "source": path.name,
                "title": title_match.group(1).strip() if title_match else path.stem,
                "text": text,
            }
        )
    logger.info("共加载文档 %d 篇，来自 %s", len(documents), root)
    return documents


def build_chunks(docs_dir: str | None = None) -> list[dict]:
    """完整流程：加载文档 -> 切分 -> 生成带 id/metadata 的 chunk 列表。"""
    chunks: list[dict] = []
    for doc in load_documents(docs_dir):
        pieces = split_text(doc["text"])
        for index, piece in enumerate(pieces):
            chunks.append(
                {
                    "id": f"{doc['source']}#{index}",
                    "text": piece,
                    "metadata": {
                        "source": doc["source"],
                        "title": doc["title"],
                        "chunk": index,
                    },
                }
            )
    logger.info("切分完成，共 %d 个 chunk", len(chunks))
    return chunks
