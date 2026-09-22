"""回归测试：核心算法单测 + Mock 模式端到端链路。

全部用例都不访问网络、不消耗 API 额度：
- 单元测试覆盖 JSON 抽取、子任务清洗、RRF 融合、分词、切分、依赖检测；
- 端到端测试用 MOCK_LLM 跑完整状态机，覆盖正常链路、依赖短路、checkpoint 落盘。

跑法：
    pytest
"""
import pytest

from agents.manager import normalize_subtasks
from app.config import settings
from app.utils import extract_json
from graph.state import failed_deps, initial_state
from rag.embedding import LocalHashEmbedding, tokenize
from rag.hybrid_search import HybridRetriever
from rag.loader import build_chunks, split_text


# 注：`isolated` fixture 已提到 conftest.py，供所有测试文件共用


# ---------------------------------------------------------------- 工具函数
@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 2}\n```', {"a": 2}),
        ('好的，结果如下：{"a": 3} 以上。', {"a": 3}),
        ('```\n{"a": {"b": 4}}\n```', {"a": {"b": 4}}),
        ('{"subtasks":[{"id":"t1","deps":[]}]}', {"subtasks": [{"id": "t1", "deps": []}]}),
    ],
)
def test_extract_json_handles_model_noise(raw, expected):
    """模型经常多写解释或包代码块，抽取必须容错。"""
    assert extract_json(raw) == expected


def test_extract_json_raises_on_garbage():
    with pytest.raises(ValueError):
        extract_json("这里完全没有 JSON")


# ---------------------------------------------------------------- 子任务清洗
def test_normalize_subtasks_cleans_model_output():
    raw = [
        {"id": "t1", "title": "A", "deps": ["t9"]},  # 依赖未定义的 id
        {"id": "t1", "title": "B", "deps": ["t1"]},  # id 重复 + 自依赖
        "这不是字典",                                    # 脏数据
        {"title": "C"},                               # 缺 id
    ]
    out = normalize_subtasks(raw)

    ids = [item["id"] for item in out]
    assert len(out) == 3, "脏数据应被丢弃"
    assert len(set(ids)) == len(ids), "id 必须唯一，否则 retries 字典会互相覆盖"
    assert out[0]["deps"] == [], "未定义的依赖必须被过滤"
    assert all(item["acceptance"] for item in out), "缺失的验收标准要有默认值"


# ---------------------------------------------------------------- RRF 融合
def test_rrf_fusion_rewards_documents_hit_by_both_lists():
    bm25 = [{"id": "x"}, {"id": "y"}, {"id": "z"}]
    vector = [{"id": "y"}, {"id": "w"}]

    fused = dict(HybridRetriever.rrf_fuse([bm25, vector], k=60))

    assert max(fused, key=fused.get) == "y", "两路都命中的文档应排第一"
    assert "w" in fused and "x" in fused
    # 融合分 = 1/(k+rank) 之和，名次越小分越高
    assert fused["y"] > fused["x"]


def test_rrf_fusion_ignores_score_scale():
    """RRF 只看名次，因此 BM25 那种无上界的大分数不会压过向量结果。"""
    huge = [{"id": "a", "score": 9999.0}, {"id": "b", "score": 1.0}]
    small = [{"id": "b", "score": 0.01}]
    fused = dict(HybridRetriever.rrf_fuse([huge, small], k=60))
    assert fused["b"] > fused["a"]


# ---------------------------------------------------------------- 检索组件
def test_tokenize_emits_chinese_bigrams():
    tokens = tokenize("断点续跑")
    assert "断点" in tokens and "续跑" in tokens, "中文需要双字切分才能匹配「续跑」这类查询"
    assert "hello" in tokenize("Hello World")


def test_local_hash_embedding_is_deterministic_and_normalized():
    embedder = LocalHashEmbedding(dim=64)
    first = embedder.embed_query("测试文本")
    second = embedder.embed_query("测试文本")

    assert first == second, "哈希向量必须可复现，否则索引每次重建都不一样"
    assert len(first) == 64
    assert abs(sum(value * value for value in first) - 1.0) < 1e-5, "L2 归一化后才能用余弦相似度"


def test_split_text_respects_chunk_size():
    text = "这是一个用于测试切分逻辑的段落。" * 200
    chunks = split_text(text, chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1, "长文本应该被切成多段"
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_corpus_is_loadable_and_splittable():
    chunks = build_chunks(settings.docs_dir)
    assert len(chunks) > 10, f"语料应至少切出十几个 chunk，实际 {len(chunks)}"
    assert all(chunk["id"] and chunk["text"] for chunk in chunks)
    assert all(chunk["metadata"].get("source") for chunk in chunks), "必须带来源，Worker 引用要用"


# ---------------------------------------------------------------- 依赖检测
def test_failed_deps_detects_failed_dependency():
    state = initial_state("测试")
    state["results"] = [{"id": "t1", "passed": False}, {"id": "t2", "passed": True}]

    assert failed_deps(state, {"id": "t3", "deps": ["t1"]}) == ["t1"]
    assert failed_deps(state, {"id": "t3", "deps": ["t2"]}) == []
    assert failed_deps(state, {"id": "t3", "deps": []}) == []


# ---------------------------------------------------------------- 端到端链路
def test_full_pipeline_finishes_in_mock_mode(isolated):
    """最小链路：Manager 拆任务 → Worker 执行 → Checker 验收 → 汇总。"""
    from graph.workflow import run_task

    state = run_task(task="验证最小链路", thread_id="pytest-main")

    assert state["status"] == "finished"
    assert len(state["subtasks"]) >= 1
    assert len(state["results"]) == len(state["subtasks"]), "每个子任务都要有验收结论"
    assert all(item["passed"] for item in state["results"])
    assert state["final_answer"], "必须产出最终交付物"


def test_checkpoint_persists_state(isolated):
    """断点续跑：状态必须能从 SQLite 读回来。"""
    from graph.workflow import get_task_state, run_task

    run_task(task="验证 checkpoint 持久化", thread_id="pytest-ckpt")
    snapshot = get_task_state("pytest-ckpt")

    assert snapshot["values"]["task"] == "验证 checkpoint 持久化"
    assert snapshot["values"]["status"] == "finished"
    assert snapshot["values"]["logs"], "执行日志应随状态一起落盘"
    assert snapshot["next"] == [], "任务结束后不应再有待执行节点"


def test_dependency_failure_short_circuits_downstream(isolated, monkeypatch):
    """依赖失败时，后续子任务应被短路，且不消耗重试次数。"""
    monkeypatch.setattr(settings, "mock_checker_fail_times", 5)  # 让 Checker 一直打回

    from graph.workflow import run_task

    state = run_task(task="验证依赖短路", thread_id="pytest-block")
    results = {item["id"]: item for item in state["results"]}

    assert state["status"] == "partial"
    # t1 走正常重试：1 次原始 + max_retry_per_subtask 次重试
    assert results["t1"]["attempts"] == settings.max_retry_per_subtask + 1
    assert results["t1"]["passed"] is False
    # t2 / t3 依赖 t1，应在调度阶段就被短路：它们从未被执行过，所以 attempts 记 0
    # （旧版靠 current_index 线性推进，必须走到 Checker 才发现依赖失败，所以那时记的是 1）
    assert results["t2"]["attempts"] == 0
    assert "依赖子任务未通过" in results["t2"]["reason"]
    assert any("因依赖" in line and "跳过" in line for line in state["logs"])


def test_retry_recovers_when_checker_rejects_once(isolated, monkeypatch):
    """被打回一次后重做成功，最终仍应 finished，且尝试次数为 2。"""
    monkeypatch.setattr(settings, "mock_checker_fail_times", 1)

    from graph.workflow import run_task

    state = run_task(task="验证重试后通过", thread_id="pytest-retry")

    assert state["status"] == "finished"
    assert all(item["attempts"] == 2 for item in state["results"])


# ---------------------------------------------------------------- 混合检索
def test_hybrid_search_recalls_relevant_document(isolated):
    """混合检索的召回质量回归：问 RRF 应该能召回 08-rrf.md。

    这条用例是后续调参（切分粒度 / RRF_K / embedding）的基准线。
    """
    from rag.hybrid_search import get_retriever

    retriever = get_retriever()
    retriever.index_chunks(build_chunks(settings.docs_dir))

    hits = retriever.search("RRF 倒数排名融合是怎么算的", top_k=3)
    sources = [hit["source"] for hit in hits]

    assert hits, "检索不应返回空结果"
    assert any("08-rrf" in source for source in sources), f"未召回 RRF 文档，实际召回：{sources}"
