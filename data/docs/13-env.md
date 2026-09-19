# 环境变量配置说明

所有配置项都可以通过 `.env` 或系统环境变量覆盖，代码里没有硬编码的密钥。

## LLM 相关

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | 空 | 留空会自动进入 Mock 模式 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | OpenAI 兼容地址 |
| `MODEL_NAME` | `deepseek-chat` | 也可换成 `deepseek-reasoner` |
| `TEMPERATURE` | `0.2` | 编排类任务建议低温度 |
| `REQUEST_TIMEOUT` | `120` | 单次请求超时秒数 |
| `MOCK_LLM` | `false` | 强制离线 Mock |

## RAG 相关

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DOCS_DIR` | `data/docs` | 语料目录 |
| `CHUNK_SIZE` | `500` | 切分粒度 |
| `CHUNK_OVERLAP` | `80` | 相邻 chunk 重叠 |
| `RAG_TOP_K` | `3` | 注入 Prompt 的片段数 |
| `RRF_K` | `60` | RRF 平滑常数 |
| `EMBEDDING_PROVIDER` | `local` | `local` 或 `openai` |
| `EMBEDDING_DIM` | `384` | 本地哈希向量维度 |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | 远程 embedding 模型名 |
| `EMBEDDING_API_KEY` | 空 | `provider=openai` 时必填 |
| `EMBEDDING_BASE_URL` | 空 | 留空则复用 `DEEPSEEK_BASE_URL` |

## 编排相关

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `SQLITE_PATH` | `data/checkpoint.sqlite` | checkpoint 位置 |
| `MAX_SUBTASKS` | `5` | 单次任务最多拆几个子任务 |
| `MAX_RETRY_PER_SUBTASK` | `2` | 单个子任务最多被打回几次 |
| `CORS_ORIGINS` | `*` | 逗号分隔的允许来源 |

## 安全提醒

`.env` 已在 `.gitignore` 与 `.dockerignore` 中排除。如果曾经误提交过 Key，请立刻去服务商后台轮换，因为 git 历史里的密钥即使删除文件也依然可被检出。
