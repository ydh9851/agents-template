"""HTTP 接口契约测试：健康检查、指标端点、鉴权开关的实际生效情况。

单测验的是函数行为，这里验的是「挂到路由上之后还对不对」——
依赖挂错地方、被中间件吞掉这类问题只有走一遍 HTTP 才看得出来。
"""
from fastapi.testclient import TestClient

from app.config import settings
from app.observability import metrics
from main import app

client = TestClient(app)


def test_health_exposes_capabilities():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_chain" in data
    assert "auth_enabled" in data
    assert "token_budget" in data


def test_metrics_endpoint_returns_prometheus_text():
    metrics.inc("test_suite_total", 1)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "test_suite_total 1" in resp.text


def test_health_stays_open_when_auth_enabled(monkeypatch):
    """健康检查必须匿名可读，否则探活也得配 Key。"""
    monkeypatch.setattr(settings, "api_keys", "secret-key")
    assert client.get("/health").status_code == 200
    assert client.get("/metrics").status_code == 200


def test_task_requires_api_key_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "api_keys", "secret-key")
    assert client.post("/task", json={"task": "写一句测试文案"}).status_code == 401

    ok = client.post(
        "/task",
        json={"task": "写一句测试文案"},
        headers={"X-API-Key": "secret-key"},
    )
    # 鉴权必须通过；任务本身成功与否不在本用例的断言范围
    assert ok.status_code != 401


def test_task_rejects_empty_task():
    resp = client.post("/task", json={"task": "   "})
    assert resp.status_code == 400


def test_stream_requires_thread_id_on_resume():
    resp = client.post("/task/stream", json={"resume": True})
    assert resp.status_code == 400
