"""鉴权与限流测试。

两者都是「默认关闭、开了才生效」，所以重点测开关边界：
关了必须放行（不能把开箱即用搞坏），开了必须拦住。
"""
import asyncio

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app import security
from app.config import settings
from app.security import SlidingWindowLimiter, rate_limit, require_api_key


def _request(client=("1.2.3.4", 1234)) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/task",
            "headers": [],
            "client": client,
        }
    )


# ---------------------------------------------------------------- 鉴权
def test_auth_disabled_allows_anyone(monkeypatch):
    monkeypatch.setattr(settings, "api_keys", "")
    asyncio.run(require_api_key(x_api_key=None, authorization=None))


def test_auth_rejects_wrong_key(monkeypatch):
    monkeypatch.setattr(settings, "api_keys", "good-key")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(require_api_key(x_api_key="bad-key", authorization=None))
    assert exc.value.status_code == 401


def test_auth_accepts_header_key(monkeypatch):
    monkeypatch.setattr(settings, "api_keys", "good-key")
    asyncio.run(require_api_key(x_api_key="good-key", authorization=None))


def test_auth_accepts_bearer_token(monkeypatch):
    monkeypatch.setattr(settings, "api_keys", "good-key")
    asyncio.run(require_api_key(x_api_key=None, authorization="Bearer good-key"))


def test_auth_supports_multiple_keys(monkeypatch):
    monkeypatch.setattr(settings, "api_keys", "k1, k2")
    asyncio.run(require_api_key(x_api_key="k2", authorization=None))


# ---------------------------------------------------------------- 限流
def test_limiter_blocks_after_limit():
    limiter = SlidingWindowLimiter(limit=2)
    assert limiter.allow("c") is True
    assert limiter.allow("c") is True
    assert limiter.allow("c") is False
    # 不同客户端互不影响
    assert limiter.allow("other") is True


def test_limiter_disabled_when_limit_is_zero():
    limiter = SlidingWindowLimiter(limit=0)
    for _ in range(50):
        assert limiter.allow("c") is True


def test_limiter_window_expires():
    # 窗口设成 0.05 秒，睡过去之后应重新放行
    limiter = SlidingWindowLimiter(limit=1, window_seconds=0.05)
    assert limiter.allow("c") is True
    assert limiter.allow("c") is False
    import time

    time.sleep(0.08)
    assert limiter.allow("c") is True


def test_rate_limit_dependency_returns_429(monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 1)
    monkeypatch.setattr(security, "limiter", SlidingWindowLimiter(limit=1))

    request = _request()
    asyncio.run(rate_limit(request))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(rate_limit(request))
    assert exc.value.status_code == 429


def test_rate_limit_disabled_by_default(monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 0)
    request = _request()
    for _ in range(5):
        asyncio.run(rate_limit(request))


# ---------------------------------------------------------------- 后端选择与降级
def test_build_limiter_defaults_to_memory(monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_backend", "memory")
    assert isinstance(security.build_limiter(), SlidingWindowLimiter)


def test_build_limiter_unknown_backend_falls_back(monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_backend", "redis-cluster-9000")
    assert isinstance(security.build_limiter(), SlidingWindowLimiter)


def test_redis_limiter_degrades_when_unreachable(monkeypatch):
    """Redis 连不上时必须退回进程内限流，而不是拒绝所有请求。

    注意这里的取舍和鉴权**相反**：
        鉴权失败要 401（宁可拒绝）；
        限流组件故障不能把整个服务挡死（宁可降级）。
    """
    limiter = security.RedisRateLimiter(limit=2)

    def boom():
        raise RuntimeError("no redis")

    monkeypatch.setattr(limiter, "_connection", boom)

    assert limiter.allow("c") is True
    assert limiter.allow("c") is True
    assert limiter.allow("c") is False  # 降级到内存滑动窗口后限流依然生效


def test_redis_limiter_disabled_when_limit_zero(monkeypatch):
    limiter = security.RedisRateLimiter(limit=0)
    for _ in range(20):
        assert limiter.allow("c") is True
