# AgentsTemplate

[![CI](https://github.com/ydh9851/agents-template/actions/workflows/ci.yml/badge.svg)](https://github.com/ydh9851/agents-template/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 项目简介

一个轻量的多 Agent 任务执行框架：丢给它一个复杂任务，它自动完成「拆解 → 执行 → 验收 → 汇总」的闭环。

- **三角色分工**：Manager 拆解任务 · Worker 执行 · Checker 独立验收，不达标就带着意见打回重做
- **技术栈**：Python 3.11+ · LangGraph 状态机 · DeepSeek（OpenAI 兼容协议）· ChromaDB + BM25 混合检索（RRF 融合）· FastAPI + SSE · SQLite checkpoint
- **开箱可跑**：无 API Key 自动进 Mock 模式，19 个 pytest 用例全部离线运行，CI 同时验证全量依赖与降级路径

> **准备继续开发？** 先读 [`docs/TARGET.md`](docs/TARGET.md) —— 它定义了项目的最终目标形态、逐条验收标准、分阶段实施路径与硬性约束。本文档说明「现在是什么样」，那份说明「要变成什么样」。

---

## 为什么需要它

把「写一份技术选型说明」这类复杂任务丢给大模型，它会一口气给你一段文本。问题是：**中间没有质检，写错的地方你只能自己发现。**

输出越长越容易出问题——漏掉要点、前后矛盾、答非所问。而这个过程中你无从干预，因为你看不到它在想什么、做到哪一步。

AgentsTemplate 的做法是**把「一次对话」拆成「一条流水线」**：

```
用户任务
   ↓
Manager   拆成子任务：描述 / 依赖 / 验收标准        ← 1 次模型调用
   ↓
Worker    按顺序执行，需要资料时调混合检索 RAG      ← N 次
   ↓
Checker   独立验收，不达标就打回重做                ← N 次
   ↓
Manager   汇总所有结果，输出最终交付物              ← 1 次
```

关键在于**执行者和验收者是分开的**。Worker 说「我做完了」不算数，Checker 会按事先定好的验收标准独立判断，不通过就带着意见打回重做。

> 没有质检的多步执行，只是把错误放大 N 倍。

## 核心特性

- **多 Agent 协作闭环** —— Manager / Worker / Checker 三角色分工，LangGraph 状态机编排，重试回环是图的一部分
- **混合检索 RAG** —— BM25（关键词）+ 向量（语义）双路召回，RRF 倒数排名融合，检索结果带来源标注
- **断点续跑** —— 每个节点执行完自动落盘 SQLite，进程崩了用同一个 `thread_id` 就能接着跑
- **SSE 流式输出** —— 节点级事件实时推送，能看到「Checker 打回了第 2 个子任务」
- **离线 Mock 模式** —— 没有 API Key 也能跑通全链路，开发与 CI 不需要真实额度
- **多级容错设计** —— 依赖缺失、外部服务异常、单点失败都有明确的降级或兜底路径

## 它是怎么工作的

### LangGraph 图结构

```
manager ──> worker ──> checker ──┬── 打回重做（未通过且未超重试上限）──> worker
   │                             │
   │                             └── 通过 / 超过重试上限 ──> 推进到下一个子任务
   │                                                              │
   └── 没有拆出子任务 ───────────────────────────────────────> summarize ──> END
```

- `manager` 入口节点：读用户任务，输出结构化子任务列表
- `worker` 执行节点：按 `current_index` 取出子任务，需要时调 RAG
- `checker` 验收节点：判定通过则推进下标；不通过则打回；超限则标记失败并继续
- `summarize` 出口节点：把所有结果汇总成最终交付物

整张图只有一条主线 + 一个重试回环，所以流程图就能读完。真正让它能跑起来的是「每个节点边界都是一个安全点」——这正是断点续跑的基础。

### 三个 Agent 的职责

| Agent | 做什么 | 不做什么 |
| --- | --- | --- |
| **Manager** | 拆解任务出 JSON；全部完成后汇总交付 | 不执行具体子任务 |
| **Worker** | 执行单个子任务；按 `need_rag` 决定是否调检索 | **不自我验收**，也不做别的子任务 |
| **Checker** | 独立判断结果是否达标；写出具体的打回理由 | 不修改结果，只判定 |

### Worker 的能力边界（重要）

Worker 是纯语言模型，**没有任何工具**：不能访问文件系统、不能读代码仓库、不能执行命令、不能联网。

它唯一的外部信息通道是 RAG 资料库。因此 Manager 被硬性约束：

- 禁止拆出「扫描仓库 / 读取源码 / 给出真实文件路径或行号 / 运行代码验证」这类子任务
- 验收标准必须能在「不访问文件系统、不执行代码」的前提下判定

这条约束是实测踩坑后补上的：早期版本里 Manager 生成了「给出真实文件路径清单」「引用具体代码位置」这类验收标准，导致 5 个子任务各重试 3 次全部失败（约 30 次无效调用）。加上约束后，同一任务改为 1 个子任务、1 次通过（4 次调用）。

**这意味着本框架目前只适合「文本产出」类任务**，接不了编码任务。想让它真正动手改文件，需要先补工具层（见「路线图」）。

## 快速开始

### 本地启动

```bash
git clone https://github.com/ydh9851/agents-template.git
cd agents-template

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

# Windows
copy .env.example .env
# macOS / Linux
cp .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY（留空则自动进入 Mock 模式）

python tools/ingest.py          # 构建 RAG 索引
uvicorn main:app --port 8000
```

打开 http://localhost:8000/health 应看到：

```json
{
  "status": "ok",
  "mock_mode": false,
  "model": "deepseek-chat",
  "index": { "backend": "chromadb", "documents": 51, "bm25_ready": true }
}
```

> 不启动服务也能用：`tools/run_task.py` 直接驱动状态机，只是少了 HTTP 那一层。

### Docker Compose

```bash
cp .env.example .env            # Windows: copy .env.example .env
docker compose up --build -d
docker compose logs -f agents-api
```

容器启动时会先执行 `python tools/ingest.py` 重建索引（幂等），再拉起服务；`./data` 已挂载为卷，checkpoint 与索引在容器重建后依然保留。

## 使用方式

### 命令行

```bash
# 跑一个任务，实时打印节点执行过程
python tools/run_task.py "说明本框架的 RRF 融合是怎么工作的"

# 断点续跑：先跑一次记下打印出的 thread_id，再用它恢复
python tools/run_task.py --thread-id <thread_id> --resume

# 完全离线跑（不消耗 API 额度）
MOCK_LLM=true python tools/run_task.py "测试任务"
```

### HTTP API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查：模型、Mock 状态、向量库后端、索引规模 |
| POST | `/task` | 执行任务，阻塞返回最终状态 |
| POST | `/task/stream` | 执行任务，SSE 流式输出 |
| GET | `/task/{thread_id}/state` | 查询某个 thread 的最新 checkpoint |
| POST | `/retrieve` | 混合检索调试 |
| POST | `/index/build` | 重建索引 |
| GET | `/index/stats` | 索引状态 |
| GET | `/` | 服务信息 |

交互式文档在 http://localhost:8000/docs （FastAPI 自动生成，可以直接在页面上调接口）。

```bash
curl -X POST http://localhost:8000/task \
  -H "Content-Type: application/json" \
  -d '{"task":"说明本框架的 RRF 融合是怎么工作的"}'
```

### SSE 流式输出

```bash
curl -N -X POST http://localhost:8000/task/stream \
  -H "Content-Type: application/json" \
  -d '{"task":"写一份技术选型说明"}'
```

事件序列：

| 事件 | 含义 | data |
| --- | --- | --- |
| `start` | 任务开始 | `thread_id`、`resume`、`task` |
| `node` | 某个节点执行完成 | `node`（节点名）、`update`（状态增量） |
| `done` | 全部结束 | 完整最终状态 |
| `error` | 执行异常 | `message` |

`node` 级事件让前端能画出实时执行流水：拆了几个子任务、正在做第几个、哪一条被打回了。

### 断点续跑

```bash
# 查询 checkpoint 状态：values 是完整状态，next 是接下来要执行的节点
curl http://localhost:8000/task/<thread_id>/state
```

任务用 `thread_id` 隔离，不同 thread 互不可见。进程重启后用同一个 `thread_id` + `resume=true` 即可从最后一个 checkpoint 继续。

## 配置

所有配置通过 `.env` 读取（pydantic-settings），代码中没有硬编码密钥。常用项：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | 空 | **留空即自动进入 Mock 模式** |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | OpenAI 兼容地址 |
| `MODEL_NAME` | `deepseek-chat` | 也可换 `deepseek-reasoner` |
| `MAX_SUBTASKS` | `5` | 单次任务最多拆几个子任务 |
| `MAX_RETRY_PER_SUBTASK` | `2` | 单个子任务最多被打回几次 |
| `RAG_TOP_K` | `3` | 注入 Prompt 的检索片段数 |
| `RRF_K` | `60` | RRF 平滑常数 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `80` | 文档切分粒度 |
| `EMBEDDING_PROVIDER` | `local` | `local`（零依赖哈希向量）/ `openai` |
| `SQLITE_PATH` | `data/checkpoint.sqlite` | checkpoint 位置 |
| `CORS_ORIGINS` | `*` | 生产环境请改成具体域名 |
| `MOCK_LLM` | `false` | 强制离线 Mock |
| `MOCK_CHECKER_FAIL_TIMES` | `0` | Mock 模式下让 Checker 前 N 次故意打回，用于验证重试分支 |

完整清单见 [`.env.example`](.env.example)。

> 改了 `.env` 必须**重启服务**才生效——配置在进程启动时读取并缓存。

## 项目结构

```
agents-template/
├── main.py                     # FastAPI 入口（HTTP + SSE）
├── app/
│   ├── config.py               # 配置层：.env -> Settings
│   ├── llm.py                  # DeepSeek 接入 + 离线 MockLLM 降级
│   ├── prompts.py              # Prompt 模板加载
│   └── utils.py                # 日志、模板渲染、稳健 JSON 抽取
├── agents/                     # 三个 Agent 的节点实现
│   ├── manager.py              #   拆任务 + 汇总
│   ├── worker.py               #   执行子任务 + 调 RAG
│   └── checker.py              #   验收 + 重试控制
├── graph/
│   ├── state.py                # AgentState 状态定义
│   └── workflow.py             # LangGraph 状态机 + 运行入口
├── rag/
│   ├── loader.py               # 文档加载与切分
│   ├── embedding.py            # Embedding 抽象（本地哈希 / OpenAI 兼容）
│   ├── store.py                # 向量库（ChromaDB / 本地 JSON 降级）
│   └── hybrid_search.py        # BM25 + 向量 + RRF 融合
├── checkpoint/
│   └── sqlite_checkpoint.py    # LangGraph SQLite checkpoint
├── prompts/                    # 4 个 Prompt 模板
├── tools/
│   ├── ingest.py               # 重建 RAG 索引
│   └── run_task.py             # 命令行跑任务
├── tests/test_smoke.py         # 回归测试（19 个用例，全部离线）
├── docs/TARGET.md              # 目标效果文档（最终目标 / 验收标准 / 实施路径）
├── data/docs/                  # RAG 语料（20 篇技术文档）
├── .github/workflows/ci.yml    # CI：全量依赖 + 降级路径双任务
├── requirements.txt / requirements-dev.txt
├── Dockerfile / docker-compose.yml
└── LICENSE
```

## 技术栈

| 层次 | 选型 |
| --- | --- |
| 运行时 | Python 3.11+（已在 3.12 验证） |
| 编排 | LangGraph 0.2.x（StateGraph + 条件边 + checkpointer） |
| LLM | DeepSeek API（OpenAI 兼容协议，经 `langchain-openai` 接入） |
| 关键词检索 | `rank_bm25`（BM25Okapi） |
| 向量库 | ChromaDB（0.5.x / 1.x 均可） |
| Embedding | 本地哈希向量（默认）/ OpenAI 兼容接口 |
| 融合算法 | RRF —— 自己实现，约 15 行 |
| 服务层 | FastAPI + Uvicorn + `sse-starlette` |
| 持久化 | SQLite（`langgraph-checkpoint-sqlite`） |
| 部署 | Docker Compose |
| 测试 | pytest（19 用例）+ GitHub Actions |

## 设计要点

### 混合检索为什么用 RRF

RRF 的公式只有一行：

```
score(d) = Σ 1 / (k + rank_i(d))
```

其中 `rank_i(d)` 是文档 d 在第 i 路结果里的名次，`k` 默认 60。

它只用名次、不用分数，因此天然免疫量纲问题——BM25 的分数是无上界实数（可能 0.3，也可能 25），余弦相似度固定在 `[-1, 1]`，直接线性加权必须先归一化，而归一化方式的选择会显著影响排序，很难调。

中文检索上，BM25 用「单字 + 相邻双字」分词（`断点续跑` → `断点 / 点续 / 续跑`），这样即使用户只搜「续跑」也能命中。

### Embedding 的现实约束

**DeepSeek 官方没有 embedding 接口。** 所以默认使用 `EMBEDDING_PROVIDER=local`：用 hashing trick 把 token 投影到 384 维并做 L2 归一化。

它的优点是零依赖、零下载、确定性可复现；缺点是**本质上是字面匹配的量化版本，没有真正的语义泛化能力**。需要「同义不同词」的召回时，把 `EMBEDDING_PROVIDER=openai` 指向任意 OpenAI 兼容的 embeddings 服务，然后重建索引。

这是本项目 RAG 部分最大的能力边界。

### 容错降级

框架在 8 个地方做了明确的容错处理，保证「依赖缺失 / 外部服务异常 / 单点失败」不会让整条链路瘫痪：

| 场景 | 行为 |
| --- | --- |
| 没有 `DEEPSEEK_API_KEY` | 自动切换 `MockLLM`，状态机真实流转，输出为占位内容 |
| 未安装 chromadb | 降级为本地 JSON 向量库（numpy 余弦相似度），接口完全一致 |
| 未安装 `langchain-text-splitters` | 使用内置的等价切分实现 |
| 未安装 `langgraph-checkpoint-sqlite` | 降级为 `MemorySaver`，日志告警 |
| RAG 检索抛异常 | 记警告并返回空资料，任务继续（检索是增强项而非必需项） |
| Checker 调用失败 / 输出无法解析 | 自动放行并留痕，避免任务永久卡在验收环节 |
| 子任务连续不达标 | 达到 `MAX_RETRY_PER_SUBTASK` 后标记未通过并继续，最终 `status=partial` |
| 子任务的依赖未通过验收 | Worker 与 Checker 双双短路，**不调用模型**，直接标记未通过并继续 |

最后一条解决的是「一处失败拖垮整条依赖链」：没有它时，一个前置子任务失败会导致后续每个子任务各重试 3 轮，全部产出同样的「无法完成」，实测浪费 20 次以上调用。

## 测试

```bash
pip install -r requirements-dev.txt

pytest          # 19 个用例，约 1 秒
pytest -v       # 看每个用例名
```

全部用例**离线运行**（走 Mock 模式，不消耗 API 额度），覆盖：

| 层次 | 覆盖内容 |
| --- | --- |
| 工具函数 | JSON 抽取容错：代码块包裹 / 夹带解释 / 嵌套结构 / 完全不是 JSON |
| 子任务清洗 | id 去重、非法依赖过滤、脏数据丢弃、缺失验收标准补默认值 |
| RRF 融合 | 双路命中的文档排第一；只看名次，不受 BM25 无上界分数影响 |
| 检索组件 | 中文双字分词、哈希向量确定性与 L2 归一化、切分长度不越界 |
| 端到端 | 最小链路 `finished`、checkpoint 落盘可查回、打回重做后恢复、依赖失败短路 |
| 检索质量回归 | 问「RRF 倒数排名融合」必须能召回 `08-rrf.md` |

CI 有两个任务：`test` 跑全量依赖（Python 3.11 + 3.12），`degraded` **刻意不装 chromadb 和 langchain-text-splitters**，用来验证上表中那两条降级路径真的可用。

### 手动验证「打回重做」

```bash
# 让 Checker 第一次故意判不通过
MOCK_CHECKER_FAIL_TIMES=1 python tools/run_task.py "验证重试"
# Windows PowerShell: $env:MOCK_CHECKER_FAIL_TIMES="1"; python tools\run_task.py "验证重试"
```

预期日志出现「未通过（第 1 次），打回 Worker 重做」，随后重新执行并通过，`尝试次数 = 2`。

把该值设为大于 `MAX_RETRY_PER_SUBTASK`，可验证「超过重试上限」：子任务标记未通过，最终 `status = partial`，任务整体不失败。

## 已知限制

这个项目是一个**轻量实现**，适合学习、二次开发和中小型文本任务。它不是生产级系统，以下差距是明确的：

**P0**

- **没有检索质量评测集** —— 目前只有一条「必须召回 `08-rrf.md`」的回归用例，无法回答「Hit@1 / MRR@5 提升了多少」。这是最优先的补强项
- **`deps` 字段尚未参与调度** —— 子任务仍按 `current_index` 顺序执行，`deps` 只用于把依赖结果注入 Worker 的 Prompt，真正的 DAG 拓扑调度与同层并行尚未实现
- **Worker 没有工具** —— 只能产出文本，接不了编码任务
- 语料仅 20 篇自建文档（51 chunk），没有来源可信度标注
- 测试覆盖了核心算法与端到端链路，但**未覆盖 FastAPI 接口层**

**P1**

- 无 metrics / traceId，多任务并发时只能靠 `thread_id` 人工对应，token 消耗不可见
- SSE 不支持断线续传（`Last-Event-ID`）与心跳，长任务在反向代理下可能被断开
- 没有并发限流与资源隔离
- 长文本在整个链路里是原文透传，未做摘要压缩，长任务上下文会膨胀

**P2**

- Manager 拆解时未引入历史对话，不支持多轮澄清
- 没有 human-in-the-loop 节点，验收失败时无法人工介入决策

**P3**

- 无 Web 前端，节点流转只能通过 SSE 或日志观察
- 失败任务不支持单个子任务重跑

## 路线图

按**性价比**排序（先做「能度量」，再做「能执行」）：

| 优先级 | 事项 | 说明 |
| --- | --- | --- |
| **1** | **检索评测集** | 从 51 个 chunk 里构造 30~50 条人工标注 query，脚本计算 Hit@1 / Hit@3 / MRR@5。有了基线，之后任何调参都能量化收益 |
| **2** | **Cross-Encoder Rerank** | 接在 RRF 之后精排，改动集中在 `rag/hybrid_search.py` + 新增 `rag/rerank.py`，投入产出比最高的效果提升 |
| **3** | **DAG 调度** | 让 `deps` 真正参与调度：拓扑排序 + 就绪集计算 + 同层并行执行 |
| **4** | **工具层 + Coder / Tester** | 给 Worker 加受控的文件读写与命令执行，框架从「文本产出」升级为「真实执行」。**必须先在 3 完成 workspace 隔离与路径校验**，否则开放写权限有越权风险 |
| **5** | **层级切分 + AutoMerging** | 面向结构化的真实领域语料（父 / 子两级切分、命中叶节点后向上合并）；语料规模上来之后才划算 |
| **6** | **上下文压缩** | 长输出摘要后再传给下游，缓解长任务的上下文膨胀与成本上涨 |

> 顺序说明：跳过度量直接堆功能，会陷入「无法判断改动好坏」的困境。

上表是**概览**。每条能力的具体验收标准、分阶段实施计划、以及「哪些事不要做」的硬性约束，见 **[`docs/TARGET.md`](docs/TARGET.md)**。

## 开发进度

- [x] FastAPI 项目初始化 + LangGraph 状态机骨架 + 接 DeepSeek
- [x] 三个 Agent：Manager 拆任务出 JSON、Worker 执行、Checker 验收
- [x] 混合检索：文档加载切分 + BM25/向量/RRF
- [x] SQLite checkpoint 断点续跑
- [x] SSE 流式输出 + Docker Compose 部署
- [x] 工程化：19 个回归用例 + GitHub Actions CI + MIT LICENSE

## 安全提醒

- `.env` 已在 `.gitignore` 与 `.dockerignore` 中排除，**不会被提交或打进镜像**
- 如果曾经误提交过真实 Key，请立刻去服务商后台轮换——git 历史里的密钥即使删除文件也依然可被检出
- 生产环境请把 `CORS_ORIGINS` 从 `*` 改成具体域名

## License

[MIT](LICENSE) © 2026 ydh9851
