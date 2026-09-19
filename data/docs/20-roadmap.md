# 目录结构与后续规划

## 目录结构

```
agents-template/
├── main.py                     # FastAPI 入口（HTTP + SSE）
├── app/
│   ├── config.py               # 配置层（.env -> Settings）
│   ├── llm.py                  # DeepSeek 接入 + 离线 Mock 降级
│   ├── prompts.py              # Prompt 模板加载
│   └── utils.py                # 日志、模板渲染、JSON 抽取
├── agents/
│   ├── manager.py              # 拆任务 + 汇总
│   ├── worker.py               # 执行子任务 + 调 RAG
│   └── checker.py              # 验收 + 重试控制
├── graph/
│   ├── state.py                # 状态定义
│   └── workflow.py             # LangGraph 状态机 + 运行入口
├── rag/
│   ├── loader.py               # 文档加载与切分
│   ├── embedding.py            # Embedding 抽象（本地哈希 / 远程）
│   ├── store.py                # 向量库（ChromaDB / 本地 JSON 降级）
│   └── hybrid_search.py        # BM25 + 向量 + RRF
├── checkpoint/
│   └── sqlite_checkpoint.py    # SQLite 断点续跑
├── prompts/                    # 四个 Prompt 模板
├── tools/
│   ├── ingest.py               # 重建索引
│   └── run_task.py             # 命令行跑任务
└── data/docs/                  # RAG 语料
```

## 已完成能力

多 Agent 协作闭环、混合检索 RAG、SQLite 断点续跑、SSE 流式输出、Docker Compose 部署、离线 Mock 模式、全链路依赖降级。

## 待改进清单

**P0（影响可用性）**

- 补充单元测试：目前只有端到端脚本，没有 pytest 用例；
- 语料规模偏小，且没有来源可信度标注。

**P1（工程化）**

- 接入指标端点与 traceId，补齐可观测性；
- SSE 支持断线续传（Last-Event-ID）与心跳；
- 并发任务的资源隔离与限流。

**P2（质量）**

- 拆任务时引入历史对话，支持多轮澄清；
- 长输出摘要压缩，控制上下文膨胀；
- 增加人工介入节点（human-in-the-loop），允许在验收失败时人工决策。

**P3（体验）**

- 提供一个最小可用的 Web 前端，可视化节点流转与子任务状态；
- 失败任务的单子任务重跑入口。
