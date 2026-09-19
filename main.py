"""FastAPI 入口：HTTP + SSE 流式输出。

启动：
    uvicorn main:app --reload --port 8000
"""
import json
from typing import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.utils import get_logger, to_jsonable
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
        "embedding_provider": settings.embedding_provider,
        "checkpoint": settings.sqlite_path,
        "index": stats,
    }


# ---------- 索引 ----------
@app.post("/index/build", summary="重建 RAG 索引")
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


@app.get("/index/stats", summary="索引状态")
def index_stats() -> dict:
    return get_retriever().stats()


@app.post("/retrieve", summary="混合检索调试（BM25 + 向量 + RRF）")
def retrieve(req: RetrieveRequest) -> dict:
    hits = get_retriever().search(req.query, top_k=req.top_k)
    return {"query": req.query, "count": len(hits), "hits": to_jsonable(hits)}


# ---------- 任务 ----------
@app.post("/task", summary="执行任务（阻塞返回最终结果）")
def create_task(req: TaskRequest) -> dict:
    if not req.resume and not (req.task or "").strip():
        raise HTTPException(status_code=400, detail="task 不能为空")
    try:
        state = run_task(task=req.task, thread_id=req.thread_id, resume=req.resume)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return to_jsonable(state)


@app.post("/task/stream", summary="执行任务（SSE 流式输出）")
def stream_task_endpoint(req: TaskRequest):
    if not req.resume and not (req.task or "").strip():
        raise HTTPException(status_code=400, detail="task 不能为空")
    if req.resume and not req.thread_id:
        raise HTTPException(status_code=400, detail="resume=true 时必须提供 thread_id")

    def events() -> Iterator[dict]:
        try:
            for name, data in stream_task(task=req.task, thread_id=req.thread_id, resume=req.resume):
                yield {"event": name, "data": json.dumps(data, ensure_ascii=False)}
        except Exception as exc:  # noqa: BLE001
            logger.exception("SSE 任务异常")
            yield {"event": "error", "data": json.dumps({"message": str(exc)}, ensure_ascii=False)}

    if EventSourceResponse is not None:
        return EventSourceResponse(events())

    # 降级：手写标准 SSE 帧
    def lines() -> Iterator[str]:
        for item in events():
            yield f"event: {item['event']}\ndata: {item['data']}\n\n"

    return StreamingResponse(
        lines(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.get("/task/{thread_id}/state", summary="查询某个 thread 的最新 checkpoint 状态")
def task_state(thread_id: str) -> dict:
    return get_task_state(thread_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=False)
