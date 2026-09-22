"""DAG 调度与同层并行的测试。

这一层决定「谁该执行」：不看下标，只看依赖。
出问题时不会报错，只会退化成「谁都等别人」或者「该并行却在串行」，所以必须钉死。
"""
from app.config import settings
from graph.scheduler import schedule
from graph.state import initial_state


def _sub(sub_id: str, deps: list[str] | None = None) -> dict:
    return {
        "id": sub_id,
        "title": sub_id,
        "description": "",
        "acceptance": "",
        "deps": deps or [],
        "need_rag": False,
    }


def _done(sub_id: str, passed: bool = True) -> dict:
    return {"id": sub_id, "title": sub_id, "output": "", "passed": passed, "reason": "", "attempts": 1}


def _state(subtasks: list[dict], results: list[dict] | None = None) -> dict:
    state = initial_state("测试任务")
    state["subtasks"] = subtasks
    state["results"] = results or []
    return state


# ---------------------------------------------------------------- 调度决策
def test_idle_when_no_subtasks():
    assert schedule(_state([])).action == "idle"


def test_idle_when_all_done():
    assert schedule(_state([_sub("t1")], [_done("t1")])).action == "idle"


def test_independent_subtasks_are_all_ready():
    plan = schedule(_state([_sub("t1"), _sub("t2"), _sub("t3")]))
    assert plan.action == "run"
    assert plan.ready == [0, 1, 2]  # 互不依赖 → 同层，一次全就绪


def test_dependent_subtask_waits_for_dependency():
    plan = schedule(_state([_sub("t1"), _sub("t2", ["t1"])]))
    assert plan.action == "run"
    assert plan.ready == [0]  # t2 还要等 t1


def test_dependent_subtask_ready_after_dependency_passed():
    plan = schedule(_state([_sub("t1"), _sub("t2", ["t1"])], [_done("t1")]))
    assert plan.action == "run"
    assert plan.ready == [1]


def test_blocked_when_dependency_failed():
    plan = schedule(_state([_sub("t1"), _sub("t2", ["t1"])], [_done("t1", passed=False)]))
    assert plan.action == "blocked"
    assert plan.blocked_ids == ["t2"]


def test_independent_branch_survives_a_failed_branch():
    """DAG 调度的关键收益：t1 失败不该挡住完全不依赖它的 t3。

    线性推进时三个子任务是排队走的，t3 只能被动等；按依赖调度后它立刻可做。
    """
    plan = schedule(
        _state(
            [_sub("t1"), _sub("t2", ["t1"]), _sub("t3")],
            [_done("t1", passed=False)],
        )
    )
    assert plan.action == "run"
    assert plan.ready == [2]  # 只有 t3 就绪，t2 被失败的依赖堵住


# ---------------------------------------------------------------- 同层并行
def test_worker_executes_whole_layer_in_one_round(isolated):
    """三个无依赖子任务应在同一轮 Worker 内全部完成。"""
    from agents.worker import worker_node

    state = _state([_sub("t1"), _sub("t2"), _sub("t3")])
    out = worker_node(state)

    assert [item["id"] for item in out["pending_results"]] == ["t1", "t2", "t3"]


def test_worker_only_runs_ready_layer(isolated):
    """有依赖关系时，一轮只做当前就绪的那一层。"""
    from agents.worker import worker_node

    state = _state([_sub("t1"), _sub("t2", ["t1"]), _sub("t3", ["t1"])])
    out = worker_node(state)

    assert [item["id"] for item in out["pending_results"]] == ["t1"]


def test_checker_verifies_whole_batch(isolated):
    from agents.checker import checker_node
    from agents.worker import worker_node

    state = _state([_sub("t1"), _sub("t2")])
    state.update(worker_node(state))
    out = checker_node(state)

    assert {item["id"] for item in out["results"]} == {"t1", "t2"}
    assert out["pending_results"] == []  # 已消费，避免下一轮重复验收


def test_full_pipeline_finishes_through_parallel_batches(isolated):
    from graph.workflow import run_task

    state = run_task(task="验证并行链路", thread_id="pytest-parallel")

    assert state["status"] == "finished"
    assert len(state["results"]) == len(state["subtasks"])
    assert all(item["passed"] for item in state["results"])


# ---------------------------------------------------------------- 打回意见传递
def test_checker_records_rejection_reason(isolated, monkeypatch):
    """被打回时必须把理由记进 last_rejection，否则重做没有依据。"""
    monkeypatch.setattr(settings, "mock_checker_fail_times", 1)

    from agents.checker import checker_node
    from agents.worker import worker_node

    state = _state([_sub("t1")])
    state.update(worker_node(state))
    out = checker_node(state)

    assert out["results"] == []                     # 打回 → 不入库
    assert out["retries"]["t1"] == 1                # 记一次重试
    assert "打回" in out["last_rejection"]["t1"]     # 理由留痕


def test_worker_carries_rejection_into_retry(isolated):
    """重做时 Worker 必须能拿到上一轮的打回意见。"""
    from agents.worker import _last_rejection

    state = _state([_sub("t1")])
    state["last_rejection"] = {"t1": "缺少资料引用，请补充来源"}

    assert _last_rejection(state, "t1") == "缺少资料引用，请补充来源"
    assert _last_rejection(state, "t2") == ""


def test_rejection_cleared_after_passing(isolated, monkeypatch):
    """通过验收后打回记录必须清掉，否则下一轮会拿旧意见误导模型。"""
    monkeypatch.setattr(settings, "mock_checker_fail_times", 1)

    from graph.workflow import run_task

    state = run_task(task="验证打回记录清理", thread_id="pytest-rejection")
    assert state["status"] == "finished"
    assert state["last_rejection"] == {}


# ---------------------------------------------------------------- 验收材料包含产物
def test_checker_sees_artifact_content(isolated, monkeypatch, tmp_path):
    """Checker 必须能看到产物文件的内容。

    否则 Worker 把成果写进文件后，正文里只剩一句「已保存到 xxx.md」，
    验收会误判成「没达到验收标准」，任务无端变成 partial（实测踩过）。
    """
    from agents.checker import _review_material
    from app import workspace

    monkeypatch.setattr(settings, "workspace_dir", str(tmp_path / "ws"))
    workspace.write_text("r.md", "RRF 融合的核心是倒数排名")

    material = _review_material("已保存到 r.md", ["r.md"])
    assert "【产物文件 r.md】" in material
    assert "RRF 融合的核心是倒数排名" in material


def test_review_material_reports_unreadable_artifact(isolated, monkeypatch, tmp_path):
    from agents.checker import _review_material

    monkeypatch.setattr(settings, "workspace_dir", str(tmp_path / "ws"))
    material = _review_material("正文", ["not-exist.md"])
    assert "读取失败" in material


def test_review_material_passthrough_without_artifacts():
    from agents.checker import _review_material

    assert _review_material("正文", []) == "正文"
    assert _review_material("正文", None) == "正文"
