"""可观测性：请求上下文 + 结构化日志 + 轻量指标。

为什么需要这一层：
    一条任务会经历 Manager → Worker（可能重试 N 次）→ Checker → 汇总，
    中间夹着若干次 LLM 调用。出问题时最想知道的三件事是：
        「哪个 thread？」「哪个节点？」「哪次调用慢 / 花了多少 token？」
    在引入这一层之前，日志是一行行散装文本：既没法按任务聚合，
    也说不清耗时和 token 到底花在哪。

三块能力，全部保持零重依赖：

    1. 上下文（contextvar）
       thread_id / node 写进上下文，所有日志自动带上，一条任务的日志能串起来。

    2. 结构化日志
       LOG_FORMAT=json 时输出「一行一个 JSON」，可直接被采集器解析；
       默认 text，本地看日志更顺眼。

    3. 指标
       手写计数器 + 观测值聚合，/metrics 暴露 Prometheus 文本格式。
       刻意不用 prometheus_client —— 这个项目的卖点是「一个文件读懂」，
       为几个指标引入一个依赖加一套全局注册表并不划算，文本格式本身很简单。
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# 1. 请求上下文
# ---------------------------------------------------------------------------
_thread_id: contextvars.ContextVar[str] = contextvars.ContextVar("thread_id", default="-")
_node: contextvars.ContextVar[str] = contextvars.ContextVar("node", default="-")


def bind_thread(thread_id: str | None) -> None:
    """绑定当前任务的 thread_id。用 contextvar 而不是全局变量：
    服务端可能并发跑多个任务，全局变量会被互相覆盖。"""
    _thread_id.set(thread_id or "-")


def get_thread_id() -> str:
    return _thread_id.get()


def get_node() -> str:
    return _node.get()


@contextmanager
def node_scope(node: str) -> Iterator[None]:
    """标记「当前在哪个节点内执行」，退出时自动还原（支持嵌套）。"""
    token = _node.set(node)
    try:
        yield
    finally:
        _node.reset(token)


# ---------------------------------------------------------------------------
# 2. 日志格式化
# ---------------------------------------------------------------------------
TEXT_FORMAT = "%(asctime)s | %(levelname)-7s | %(thread_id)-16s | %(node)-10s | %(name)s | %(message)s"


class ContextFilter(logging.Filter):
    """把上下文里的 thread_id / node 注入到每条日志记录，供 Formatter 引用。"""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "thread_id"):
            record.thread_id = get_thread_id()
        if not hasattr(record, "node"):
            record.node = get_node()
        return True


class JsonFormatter(logging.Formatter):
    """一行一个 JSON，字段固定，便于采集与检索。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "thread_id": getattr(record, "thread_id", "-"),
            "node": getattr(record, "node", "-"),
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    """配置根日志（幂等）。子 logger 通过 propagate 复用这里的 handler。"""
    global _configured
    if _configured:
        return

    try:
        from app.config import settings

        fmt = (settings.log_format or "text").strip().lower()
    except Exception:  # noqa: BLE001 - 配置还没就绪时退回默认，日志不能反过来把启动搞挂
        fmt = "text"

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(ContextFilter())
    handler.setFormatter(JsonFormatter() if fmt == "json" else logging.Formatter(TEXT_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # uvicorn 自带的 access 日志会绕开 root handler 单独输出，统一交给 root，避免双份
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).handlers.clear()
        logging.getLogger(name).propagate = True

    # 第三方库每发一次 HTTP 请求就打一行 INFO，会把业务日志整段淹掉，压到 WARNING
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str = "agents") -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# 3. 指标
# ---------------------------------------------------------------------------
class Metrics:
    """极简指标收集器：计数器 + 观测值（sum / count / max）线程安全。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple], float] = defaultdict(float)
        self._obs_sum: dict[tuple[str, tuple], float] = defaultdict(float)
        self._obs_count: dict[tuple[str, tuple], int] = defaultdict(int)
        self._obs_max: dict[tuple[str, tuple], float] = defaultdict(float)

    # ---- 写入 ----
    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self._counters[key] += value

    def observe(self, name: str, value: float, **labels: str) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self._obs_sum[key] += value
            self._obs_count[key] += 1
            self._obs_max[key] = max(self._obs_max[key], value)

    # ---- 读取 ----
    @staticmethod
    def _label_str(labels: tuple) -> str:
        if not labels:
            return ""
        inner = ",".join(f'{k}="{v}"' for k, v in labels)
        return "{" + inner + "}"

    def snapshot(self) -> dict[str, Any]:
        """给 /health 之类的人读接口用。"""
        with self._lock:
            counters = {f"{name}{self._label_str(labels)}": value
                        for (name, labels), value in self._counters.items()}
            observations = {
                f"{name}{self._label_str(labels)}": {
                    "sum": self._obs_sum[(name, labels)],
                    "count": self._obs_count[(name, labels)],
                    "max": self._obs_max[(name, labels)],
                }
                for (name, labels) in self._obs_sum
            }
        return {"counters": counters, "observations": observations}

    def render(self) -> str:
        """Prometheus 文本格式（ exposition format ）。"""
        with self._lock:
            counters = dict(self._counters)
            obs_sum = dict(self._obs_sum)
            obs_count = dict(self._obs_count)
            obs_max = dict(self._obs_max)

        lines: list[str] = []
        declared: set[str] = set()

        for (name, labels), value in sorted(counters.items()):
            if name not in declared:
                lines.append(f"# TYPE {name} counter")
                declared.add(name)
            lines.append(f"{name}{self._label_str(labels)} {value:g}")

        for name in sorted({n for n, _ in obs_sum}):
            if name not in declared:
                # 简化成 summary 的三个序列，够看耗时分布与峰值
                lines.append(f"# TYPE {name} summary")
                declared.add(name)
            for (n, labels), total in sorted(obs_sum.items()):
                if n != name:
                    continue
                label_str = self._label_str(labels)
                lines.append(f"{name}_sum{label_str} {total:g}")
                lines.append(f"{name}_count{label_str} {obs_count[(n, labels)]}")
                lines.append(f"{name}_max{label_str} {obs_max[(n, labels)]:g}")

        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        """测试隔离用。"""
        with self._lock:
            self._counters.clear()
            self._obs_sum.clear()
            self._obs_count.clear()
            self._obs_max.clear()


metrics = Metrics()
