"""重建 RAG 索引。

用法：
    python tools/ingest.py
    python tools/ingest.py --docs-dir data/docs --no-reset
"""
import argparse
import sys
from pathlib import Path

# 允许直接以脚本方式运行（python tools/ingest.py）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.utils import get_logger  # noqa: E402
from rag.hybrid_search import get_retriever, reset_retriever  # noqa: E402
from rag.loader import build_chunks  # noqa: E402
from rag.store import get_vector_store  # noqa: E402

logger = get_logger("tools.ingest")


def main() -> int:
    parser = argparse.ArgumentParser(description="重建混合检索索引（向量库 + BM25）")
    parser.add_argument("--docs-dir", default=settings.docs_dir, help="文档目录，默认 data/docs")
    parser.add_argument("--no-reset", action="store_true", help="增量追加，不清空旧索引")
    args = parser.parse_args()

    chunks = build_chunks(args.docs_dir)
    if not chunks:
        logger.error("没有加载到任何文档，请检查目录：%s", args.docs_dir)
        return 1

    if not args.no_reset:
        get_vector_store().reset()
        reset_retriever()

    retriever = get_retriever()
    count = retriever.index_chunks(chunks)

    logger.info("索引完成：写入 %d 个 chunk", count)
    logger.info("索引状态：%s", retriever.stats())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
