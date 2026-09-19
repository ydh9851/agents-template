# 部署：本地与 Docker

## 本地启动

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env      # Windows: copy .env.example .env
python tools/ingest.py    # 构建 RAG 索引
uvicorn main:app --reload --port 8000
```

## Docker Compose 启动

```bash
cp .env.example .env
docker compose up --build -d
docker compose logs -f agents-api
```

`docker-compose.yml` 做了三件关键的事：

1. `env_file: .env` 注入密钥，**不打进镜像**（`.dockerignore` 里已排除 `.env`）；
2. 把 `./data` 挂载到 `/app/data`，checkpoint 与索引在容器重建后依然保留；
3. 覆盖 `DOCS_DIR` / `SQLITE_PATH` / `CHROMA_DIR` 为容器内路径，避免 `.env` 里的本地绝对路径污染容器环境。

## 容器启动流程

Dockerfile 的 CMD 是 `python tools/ingest.py && uvicorn main:app ...`，即**先重建索引再拉起服务**。`tools/ingest.py` 是幂等的，重复执行只会覆盖索引。

## 健康检查

`GET /health` 返回服务状态、当前模型、是否 Mock 模式、向量库后端与索引文档数。Dockerfile 与 compose 都配置了这个探针。

## 生产环境注意

- 把 `CORS_ORIGINS` 从 `*` 改成具体域名；
- 用 Nginx / Caddy 做 TLS 终结，并确保对 SSE 路径关闭响应缓冲；
- `uvicorn` 单进程足够本项目规模，需要横向扩展时应把 checkpoint 换成 Postgres 实现（LangGraph 提供 `PostgresSaver`）；
- 镜像里包含 chromadb 与 numpy，体积约 1GB 量级，CI 建议用构建缓存。
