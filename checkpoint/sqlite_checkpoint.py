"""LangGraph 断点续跑：SQLite checkpoint。

用官方 SqliteSaver（langgraph-checkpoint-sqlite），每个节点执行完自动落盘，
服务重启后用同一个 thread_id 即可从最后一个 checkpoint 继续。
如果环境里没装该包，则降级为内存 checkpointer（进程内可续跑，重启丢失）。
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.config import settings
from app.utils import get_logger

logger = get_logger("checkpoint")


@contextmanager
def open_checkpointer(path: str | None = None) -> Iterator[object]:
    """打开一个 checkpointer。用法：with open_checkpointer() as cp: ..."""
    db_path = Path(path or settings.sqlite_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        from langgraph.checkpoint.memory import MemorySaver

        logger.warning("未安装 langgraph-checkpoint-sqlite，降级为内存 checkpoint（重启后无法续跑）")
        yield MemorySaver()
        return

    # check_same_thread=False：FastAPI 在线程池里跑同步图，连接需要跨线程使用
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        saver.setup()
        logger.info("checkpoint 已就绪：%s", db_path)
        yield saver
    finally:
        conn.close()


def checkpoint_path() -> str:
    return settings.sqlite_path
