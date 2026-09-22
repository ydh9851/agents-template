"""鉴权与限流（都可开关，默认关闭）。

为什么默认关闭：
    这个项目的原则是「clone 下来就能跑」。一上来强制配 Key 会劝退使用者，
    所以默认不开启；生产部署时设置 API_KEYS 即自动生效。

    鉴权：请求头 X-API-Key，或 Authorization: Bearer <key>
    限流：内存滑动窗口，按客户端 IP 计数，超限返回 429

为什么限流用「依赖」而不是「中间件」：
    Starlette 的 BaseHTTPMiddleware 会把下游包进独立任务，对 SSE 长连接不友好
    （历史上出现过流式响应被缓冲、边生成边推送失效的问题）。
    用 FastAPI 依赖实现既能拿到 Request，又完全不干扰流式输出。
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Any

from fastapi import Header, HTTPException, Request

from app.config import settings
from app.observability import get_logger, metrics

logger = get_logger("app.security")


class SlidingWindowLimiter:
    """内存滑动窗口限流。

    单进程有效。多副本部署时需要换成 Redis 计数 —— 那是另一个量级的复杂度，
    这个项目的定位用不上，所以这里留一句注释，而不是塞一个「看起来分布式、
    实际仍按单进程算」的假实现。
    """

    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window = window_seconds
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        if self.limit <= 0:
            return True
        now = time.time()
        with self._lock:
            bucket = self._hits[key]
            while bucket and now - bucket[0] > self.window:
                bucket.popleft()
            if len(bucket) >= self.limit:
                return False
            bucket.append(now)
            return True

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


def _client_key(request: Request) -> str:
    """取客户端标识。开了 TRUST_PROXY 才认 X-Forwarded-For，
    否则任何人都能伪造这个头绕过限流。"""
    if settings.trust_proxy:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RedisRateLimiter:
    """基于 Redis 的固定窗口计数，多个副本共享同一份配额。

    为什么用固定窗口而不是滑动窗口：
        固定窗口一条 INCR 就够；滑动窗口要 ZSET 或 Lua 脚本。
        对「每分钟 N 次」这种粗粒度限流，窗口边界上的少量误差完全可以接受。

    可用性优先的降级：
        Redis 连不上时**不拒绝请求**，而是退回进程内限流 ——
        限流组件故障不应该把整个服务挡死，这是和「鉴权失败必须 401」相反的取舍。
    """

    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window = int(window_seconds)
        self._client: Any = None
        self._fallback = SlidingWindowLimiter(limit, window_seconds)

    def _connection(self) -> Any:
        if self._client is None:
            import redis  # 可选依赖：没装就永远走降级分支

            client = redis.Redis.from_url(
                settings.redis_url, decode_responses=True, socket_timeout=1.0
            )
            client.ping()  # 提前探活，避免第一次请求才发现连不上
            self._client = client
        return self._client

    def allow(self, key: str) -> bool:
        if self.limit <= 0:
            return True
        # 按「第几个时间窗」分桶，窗口一换计数器自然重置
        bucket = f"ratelimit:{key}:{int(time.time() // self.window)}"
        try:
            client = self._connection()
            count = client.incr(bucket)
            if count == 1:
                client.expire(bucket, self.window + 1)
            return int(count) <= self.limit
        except Exception as exc:  # noqa: BLE001 - 连不上/超时/没装包都在这里兜住
            logger.warning("Redis 限流不可用（%s），本次降级为进程内限流", exc)
            self._client = None  # 下次请求时重连
            return self._fallback.allow(key)

    def reset(self) -> None:
        self._fallback.reset()


def build_limiter() -> Any:
    """按配置构造限流器。未知后端或 redis 不可用时一律退回内存实现。"""
    backend = (settings.rate_limit_backend or "memory").strip().lower()
    limit = int(settings.rate_limit_per_minute)
    if backend == "redis":
        return RedisRateLimiter(limit)
    return SlidingWindowLimiter(limit)


# 进程级单例；测试里改 limit 用 reset_limiter()
limiter = build_limiter()


def reset_limiter() -> None:
    global limiter
    limiter = build_limiter()
    limiter.reset()


async def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> None:
    """校验 API Key。未配置 API_KEYS 时直接放行（保持开箱即用）。"""
    if not settings.auth_enabled:
        return

    provided = (x_api_key or "").strip()
    if not provided and authorization and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()

    if provided not in settings.api_key_set:
        metrics.inc("auth_rejected_total")
        logger.warning("鉴权失败：missing_or_invalid_key")
        raise HTTPException(status_code=401, detail="缺少或无效的 API Key")


async def rate_limit(request: Request) -> None:
    """按客户端限流。RATE_LIMIT_PER_MINUTE=0 时不限制。"""
    limit = int(settings.rate_limit_per_minute)
    if limit <= 0:
        return

    client = _client_key(request)
    if not limiter.allow(client):
        metrics.inc("rate_limited_total")
        logger.warning("限流触发：client=%s path=%s", client, request.url.path)
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")
