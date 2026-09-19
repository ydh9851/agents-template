FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

WORKDIR /app

# 构建 chromadb / numpy 需要编译工具链
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖，利用镜像层缓存
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# 容器内固定使用容器内路径（不依赖 .env 里的本地路径）
ENV HOST=0.0.0.0 \
    PORT=8000 \
    DOCS_DIR=/app/data/docs \
    SQLITE_PATH=/app/data/checkpoint.sqlite \
    CHROMA_DIR=/app/data/chroma \
    INDEX_DIR=/app/data/index

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# 启动前先重建一次索引（幂等），再拉起服务
CMD ["sh", "-c", "python tools/ingest.py && uvicorn main:app --host 0.0.0.0 --port 8000"]
