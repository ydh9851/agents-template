"""pytest 根配置。

两件事：
1. 确保项目根目录在 sys.path 中，让 tests/ 能 import app / agents / graph；
2. 强制 MOCK_LLM=true —— 本地 .env 里通常配着真实的 DEEPSEEK_API_KEY，
   一旦某个用例意外走到真实模型，既烧额度、又依赖网络，结果还不稳定。
   环境变量的优先级高于 .env，所以在这里设一次就能兜住全部用例
   （需要跑真实模型的评估见 evals/run_eval.py）。
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MOCK_LLM", "true")


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """把运行时产物全部指向临时目录，并强制 Mock 模式，保证测试互不干扰。

    放在 conftest 里让所有测试文件共用：调度、并行、评估都依赖同一套隔离语义。
    """
    from app.config import settings
    from rag.hybrid_search import reset_retriever
    from rag.store import reset_vector_store

    monkeypatch.setattr(settings, "mock_llm", True)
    monkeypatch.setattr(settings, "mock_checker_fail_times", 0)
    monkeypatch.setattr(settings, "sqlite_path", str(tmp_path / "checkpoint.sqlite"))
    monkeypatch.setattr(settings, "chroma_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(settings, "index_dir", str(tmp_path / "index"))

    reset_vector_store()
    reset_retriever()
    yield
    reset_vector_store()
    reset_retriever()
