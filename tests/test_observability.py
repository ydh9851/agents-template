"""可观测性测试：上下文绑定、日志格式、指标渲染。

这几样东西出问题时都不会报错，只会「悄悄失效」（日志没有 trace、指标永远是 0），
所以必须由测试盯着。
"""
import json
import logging

from app.observability import JsonFormatter, Metrics, bind_thread, get_node, get_thread_id, node_scope


def test_thread_id_binds_and_reads():
    bind_thread("abc123")
    assert get_thread_id() == "abc123"


def test_thread_id_falls_back_to_dash():
    bind_thread(None)
    assert get_thread_id() == "-"


def test_node_scope_restores_previous_value():
    bind_thread("t-1")
    assert get_node() == "-"
    with node_scope("worker"):
        assert get_node() == "worker"
        with node_scope("checker"):
            assert get_node() == "checker"
        assert get_node() == "worker"
    # 退出后必须还原，否则后续日志会挂着上一个节点的标签
    assert get_node() == "-"


def test_json_formatter_outputs_parseable_line():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    record.thread_id = "t-9"
    record.node = "manager"

    payload = json.loads(formatter.format(record))
    assert payload["level"] == "INFO"
    assert payload["msg"] == "hello world"
    assert payload["thread_id"] == "t-9"
    assert payload["node"] == "manager"


def test_metrics_counter_and_observation_render():
    m = Metrics()
    m.inc("calls_total", model="a")
    m.inc("calls_total", 2, model="a")
    m.observe("latency_seconds", 0.5, model="a")
    m.observe("latency_seconds", 1.5, model="a")

    text = m.render()
    assert "# TYPE calls_total counter" in text
    assert 'calls_total{model="a"} 3' in text
    # summary 的三个序列：总和 / 次数 / 峰值
    assert 'latency_seconds_sum{model="a"} 2' in text
    assert 'latency_seconds_count{model="a"} 2' in text
    assert 'latency_seconds_max{model="a"} 1.5' in text


def test_metrics_labels_are_kept_separate():
    m = Metrics()
    m.inc("x_total", status="ok")
    m.inc("x_total", status="error")
    m.inc("x_total", status="error")
    text = m.render()
    assert 'x_total{status="ok"} 1' in text
    assert 'x_total{status="error"} 2' in text


def test_metrics_snapshot_and_reset():
    m = Metrics()
    m.inc("y_total")
    assert m.snapshot()["counters"]["y_total"] == 1
    m.reset()
    assert m.snapshot()["counters"] == {}
