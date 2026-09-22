"""FastAPI 入口：HTTP + SSE 流式输出。

启动：
    uvicorn main:app --reload --port 8000
"""
import json
import uuid
from typing import Iterator

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.observability import get_logger, metrics
from app.security import rate_limit, require_api_key
from app.task_runner import parse_last_event_id, task_registry
from app.utils import to_jsonable
from graph.workflow import get_task_state, run_task, stream_task
from rag.hybrid_search import get_retriever, reset_retriever
from rag.loader import build_chunks
from rag.store import get_vector_store

try:  # 装了 sse-starlette 就用它，没装也能靠 StreamingResponse 输出标准 SSE
    from sse_starlette.sse import EventSourceResponse
except ImportError:  # pragma: no cover
    EventSourceResponse = None

logger = get_logger("api")

app = FastAPI(
    title="AgentsTemplate",
    description="多 Agent 任务执行框架（Manager / Worker / Checker + 混合检索 RAG + 断点续跑）",
    version="1.0.0",
)

# allow_credentials 保持 False：默认 CORS_ORIGINS=*，带 Cookie 的 * 是不合法的组合
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 需要保护的路由统一挂这两个依赖。
# /、/health、/metrics 刻意不挂：健康检查与监控指标应当匿名可读，
# 否则每次探活都要配 Key，运维会骂人。
protected = [Depends(require_api_key), Depends(rate_limit)]


# ---------- 请求体 ----------
class TaskRequest(BaseModel):
    task: str | None = Field(default=None, description="用户任务；resume=true 时可为空")
    thread_id: str | None = Field(default=None, description="会话 id，用于断点续跑；不传则自动生成")
    resume: bool = Field(default=False, description="true 表示从该 thread_id 的最后一个 checkpoint 继续")


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1, le=20)


class IndexRequest(BaseModel):
    reset: bool = Field(default=True, description="是否清空旧索引后重建")


# ---------- 基础接口 ----------
@app.get("/", summary="服务信息")
def index() -> dict:
    return {
        "service": "AgentsTemplate",
        "version": "1.0.0",
        "model": settings.model_name,
        "mock_mode": settings.use_mock,
        "endpoints": ["/health", "/index/build", "/index/stats", "/retrieve", "/task", "/task/stream", "/task/{thread_id}/state"],
    }


@app.get("/health", summary="健康检查")
def health() -> dict:
    try:
        stats = get_retriever().stats()
    except Exception as exc:  # noqa: BLE001
        stats = {"error": str(exc)}
    return {
        "status": "ok",
        "mock_mode": settings.use_mock,
        "model": settings.model_name,
        "model_chain": settings.fallback_model_list,
        "embedding_provider": settings.embedding_provider,
        "auth_enabled": settings.auth_enabled,
        "rate_limit_per_minute": settings.rate_limit_per_minute,
        "token_budget": settings.task_token_budget,
        "checkpoint": settings.sqlite_path,
        "index": stats,
    }


@app.get("/metrics", summary="Prometheus 指标", response_class=PlainTextResponse)
def metrics_endpoint() -> str:
    """暴露 Prometheus 文本格式指标：LLM 调用次数 / 耗时 / token、任务状态分布。

    刻意不加鉴权：监控系统通常独立部署，要求它带业务 Key 只会让接入变麻烦；
    真要保护应该交给网关或反向代理。
    """
    return metrics.render()


# ---------- 索引 ----------
@app.post("/index/build", dependencies=protected, summary="重建 RAG 索引")
def build_index(req: IndexRequest | None = None) -> dict:
    reset = True if req is None else req.reset
    chunks = build_chunks(settings.docs_dir)
    if not chunks:
        raise HTTPException(status_code=400, detail=f"没有加载到任何文档，请检查 {settings.docs_dir}")

    store = get_vector_store()
    if reset:
        store.reset()
        reset_retriever()

    count = get_retriever().index_chunks(chunks)
    logger.info("索引重建完成，共 %d 个 chunk", count)
    return {"indexed_chunks": count, **get_retriever().stats()}


@app.get("/index/stats", dependencies=protected, summary="索引状态")
def index_stats() -> dict:
    return get_retriever().stats()


@app.post("/retrieve", dependencies=protected, summary="混合检索调试（BM25 + 向量 + RRF）")
def retrieve(req: RetrieveRequest) -> dict:
    hits = get_retriever().search(req.query, top_k=req.top_k)
    return {"query": req.query, "count": len(hits), "hits": to_jsonable(hits)}


# ---------- 任务 ----------
@app.post("/task", dependencies=protected, summary="执行任务（阻塞返回最终结果）")
def create_task(req: TaskRequest) -> dict:
    if not req.resume and not (req.task or "").strip():
        raise HTTPException(status_code=400, detail="task 不能为空")
    try:
        state = run_task(task=req.task, thread_id=req.thread_id, resume=req.resume)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return to_jsonable(state)


@app.post("/task/stream", dependencies=protected, summary="执行任务（SSE 流式输出，支持断线续传）")
def stream_task_endpoint(
    req: TaskRequest,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
):
    """执行任务并流式推送事件。

    任务跑在后台线程里，SSE 连接只是事件的读者 —— **连接断开不会中止任务**。
    客户端重连时带上同一个 `thread_id` 与 `Last-Event-ID` 请求头，
    服务端就从该 id 之后继续推送，不会重复也不丢事件。

    浏览器原生 `EventSource` 重连时会自动带 `Last-Event-ID`；
    用 fetch 自读流（像本项目前端那样）需要手动带上。
    """
    thread_id = (req.thread_id or "").strip()
    known = task_registry.get(thread_id) if thread_id else None

    if known is None:
        # 只有「全新任务」才校验入参：续传时请求体里的 task 可以为空
        if req.resume and not thread_id:
            raise HTTPException(status_code=400, detail="resume=true 时必须提供 thread_id")
        if not req.resume and not (req.task or "").strip():
            raise HTTPException(status_code=400, detail="task 不能为空")
        if not thread_id:
            thread_id = uuid.uuid4().hex

    from_event = parse_last_event_id(last_event_id)

    def events() -> Iterator[dict]:
        run = task_registry.start(thread_id, req.task, req.resume, stream_task)
        cursor = from_event
        while True:
            batch = run.wait_for(cursor, timeout=1.0)
            for event in batch:
                cursor = event.id
                yield {
                    "id": str(event.id),
                    "event": event.name,
                    "data": json.dumps(event.data, ensure_ascii=False),
                }
            # 任务已结束且缓冲区里没有新事件 → 收尾
            if run.finished and not batch:
                break

    if EventSourceResponse is not None:
        # ping 让空闲连接也有字节流动，避免被反向代理当死连接掐掉
        return EventSourceResponse(events(), ping=15)

    # 降级：手写标准 SSE 帧（必须带 id:，否则客户端无法续传）
    def lines() -> Iterator[str]:
        for item in events():
            yield f"id: {item['id']}\nevent: {item['event']}\ndata: {item['data']}\n\n"

    return StreamingResponse(
        lines(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.get("/task/{thread_id}/state", dependencies=protected, summary="查询某个 thread 的最新 checkpoint 状态")
def task_state(thread_id: str) -> dict:
    return get_task_state(thread_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=False)
