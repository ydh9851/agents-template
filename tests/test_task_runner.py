"""后台执行与事件缓冲的测试（SSE 断线续传的基础）。

核心要验证的是两件在真实网络里很难复现、但一旦出问题就很致命的事：

1. **读的人断线，任务照跑**（事件不丢）
2. **重连能从断点接着读**（不重复、不缺失）

这两点必须在单元层面钉死，否则「断线续传」只是一句宣传语。
"""
import threading
import time

from app.task_runner import TaskRegistry, TaskRun, parse_last_event_id


def _runner(events, delay: float = 0.0):
    """构造一个按顺序吐事件的假执行器。"""

    def run(task, thread_id, resume):
        for name, data in events:
            if delay:
                time.sleep(delay)
            yield name, data

    return run


# ---------------------------------------------------------------- 事件缓冲
def test_event_ids_are_monotonic():
    run = TaskRun(thread_id="t", task="x", resume=False)
    run.append("start", {"a": 1})
    run.append("node", {"b": 2})

    events = run.wait_for(-1)
    assert [e.id for e in events] == [0, 1]
    assert [e.name for e in events] == ["start", "node"]


def test_wait_for_returns_only_new_events():
    run = TaskRun(thread_id="t", task="x", resume=False)
    run.append("start", {})
    run.append("node", {})

    assert [e.id for e in run.wait_for(0)] == [1]
    assert run.wait_for(1) == []


def test_wait_for_wakes_up_on_new_event():
    """没有新事件时应阻塞等待，且被 append 立刻唤醒（而不是等满超时）。"""
    run = TaskRun(thread_id="t", task="x", resume=False)

    def produce():
        time.sleep(0.05)
        run.append("node", {"late": True})

    threading.Thread(target=produce, daemon=True).start()
    started = time.perf_counter()
    events = run.wait_for(-1, timeout=3.0)

    assert len(events) == 1
    assert time.perf_counter() - started < 1.5


def test_finished_run_returns_immediately():
    run = TaskRun(thread_id="t", task="x", resume=False)
    run.mark_finished()

    started = time.perf_counter()
    assert run.wait_for(-1, timeout=5.0) == []
    assert time.perf_counter() - started < 0.5


# ---------------------------------------------------------------- 断线续传的核心保证
def test_reader_disconnect_does_not_lose_events():
    """读者中途『断线』（停止读取）期间任务继续跑，重连后事件一条不少。"""
    gate = threading.Event()

    def runner(task, thread_id, resume):
        yield "start", {}
        gate.wait(3)  # 卡住，模拟「这时客户端断线了」
        yield "node", {"i": 1}
        yield "done", {}

    registry = TaskRegistry()
    run = registry.start("t-disconnect", "x", False, runner)

    # 第一次连接：读到 start 就断开
    first = run.wait_for(-1, timeout=3.0)
    assert [e.name for e in first] == ["start"]

    # 断线期间任务继续跑完
    gate.set()

    # 重连：带着最后收到的 id 接着读
    seen = list(first)
    cursor = first[-1].id
    deadline = time.time() + 5
    while time.time() < deadline:
        batch = run.wait_for(cursor, timeout=0.5)
        for event in batch:
            cursor = event.id
            seen.append(event)
        if run.finished and not batch:
            break

    assert [e.name for e in seen] == ["start", "node", "done"]
    assert [e.id for e in seen] == [0, 1, 2]  # 不重复、不缺失


# ---------------------------------------------------------------- 登记表
def test_registry_reuses_running_run():
    """同一个 thread_id 重连时必须拿到同一个 TaskRun，否则就是重跑而不是续传。"""
    registry = TaskRegistry()
    gate = threading.Event()

    def runner(task, thread_id, resume):
        gate.wait(3)
        yield "done", {}

    first = registry.start("t-reuse", "x", False, runner)
    time.sleep(0.05)
    second = registry.start("t-reuse", "x", False, runner)

    assert first is second
    gate.set()


def test_registry_reuses_finished_run_for_replay():
    """已结束的执行也要能被复用 —— 重连时要能重放，而不是重新跑一遍。"""
    registry = TaskRegistry()
    runner = _runner([("done", {})])

    first = registry.start("t-replay", "x", False, runner)
    deadline = time.time() + 3
    while not first.finished and time.time() < deadline:
        time.sleep(0.01)

    assert registry.start("t-replay", "x", False, runner) is first


def test_registry_isolates_different_threads():
    registry = TaskRegistry()
    gate = threading.Event()

    def runner(task, thread_id, resume):
        gate.wait(3)
        yield "done", {}

    assert registry.start("ta", "x", False, runner) is not registry.start("tb", "x", False, runner)
    gate.set()


def test_registry_evicts_old_runs():
    registry = TaskRegistry(max_runs=2)
    runner = _runner([("done", {})])

    for index in range(5):
        run = registry.start(f"t-{index}", "x", False, runner)
        deadline = time.time() + 2
        while not run.finished and time.time() < deadline:
            time.sleep(0.005)

    assert registry.size <= 2


def test_runner_error_becomes_event():
    """后台线程里的异常不能悄悄消失，必须变成 error 事件让客户端看到。"""

    def broken(task, thread_id, resume):
        yield "start", {}
        raise RuntimeError("模拟执行失败")

    registry = TaskRegistry()
    run = registry.start("t-error", "x", False, broken)

    deadline = time.time() + 3
    while not run.finished and time.time() < deadline:
        time.sleep(0.01)

    events = run.wait_for(-1)
    assert [e.name for e in events] == ["start", "error"]
    assert "模拟执行失败" in events[-1].data["message"]


# ---------------------------------------------------------------- Last-Event-ID 解析
def test_parse_last_event_id():
    assert parse_last_event_id(None) == -1
    assert parse_last_event_id("") == -1
    assert parse_last_event_id("3") == 3
    assert parse_last_event_id(" 7 ") == 7
    assert parse_last_event_id("not-a-number") == -1


# ---------------------------------------------------------------- HTTP 契约
def test_stream_endpoint_emits_event_ids(isolated):
    """SSE 帧必须带 id:，否则客户端根本没有续传的依据。"""
    from fastapi.testclient import TestClient

    from main import app

    client = TestClient(app)
    with client.stream(
        "POST", "/task/stream", json={"task": "写一句测试文案", "thread_id": "sse-ids"}
    ) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())

    assert "id:" in body
    assert "event: start" in body
    assert "event: done" in body


def test_stream_endpoint_replays_after_last_event_id(isolated, monkeypatch):
    """带 Last-Event-ID 重连时，已收到的事件不应重复推送。

    这里刻意强制走「手写 SSE 帧」的降级分支：sse_starlette 内部有个全局的
    should_exit_event，在同一进程里发第二次流式请求时会绑定到另一个事件循环而报错
    （它自身的限制，与本项目代码无关）。两条分支共用同一个 events() 生成器，
    所以这里验证的续传语义对两者都成立。
    """
    import main
    from fastapi.testclient import TestClient

    monkeypatch.setattr(main, "EventSourceResponse", None)

    client = TestClient(main.app)
    payload = {"task": "写一句测试文案", "thread_id": "sse-resume"}

    with client.stream("POST", "/task/stream", json=payload) as resp:
        assert "event: start" in "".join(resp.iter_text())

    # 同一个 thread_id 重连，声明「我已经收到 id=0 了」
    with client.stream(
        "POST", "/task/stream", json=payload, headers={"Last-Event-ID": "0"}
    ) as resp:
        second = "".join(resp.iter_text())

    assert "id: 0" not in second          # 已收到的不再重放
    assert "id: 1" in second              # 之后的照常推送
