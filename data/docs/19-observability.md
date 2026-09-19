# 日志与可观测性

## 日志层次

框架用统一的 `get_logger` 输出带时间戳、级别和模块名的结构化日志：

```
2026-09-19 10:20:31 | INFO    | graph.workflow | 节点执行完成：manager
2026-09-19 10:20:45 | INFO    | agent.worker   | Worker 执行子任务 t2（第 1 次尝试）
2026-09-19 10:20:58 | INFO    | agent.checker  | Checker 子任务 t2 判定：通过
```

关键节点都有日志：Manager 拆出几个子任务、Worker 第几次尝试、是否命中检索、Checker 判定结果与打回原因。

## 状态里的 logs 字段

除了运行日志，`AgentState.logs` 里还保存了一份人类可读的执行轨迹。它会随着 checkpoint 一起落盘，所以 `GET /task/{thread_id}/state` 能把整条轨迹取回来——这在排查「为什么这个子任务被反复打回」时特别有用。

## 三处可观测入口

| 入口 | 用途 |
| --- | --- |
| `GET /health` | 服务、模型、Mock 状态、向量库后端、索引文档数 |
| `GET /index/stats` | 索引规模与 BM25 是否就绪 |
| `POST /retrieve` | 单次检索的完整命中结果与融合分 |

## 当前不足

- 没有接入 Prometheus / OpenMetrics，缺少 actuator 式的指标端点；
- 日志里**没有 traceId**，多任务并发时只能靠 `thread_id` 人工对应；
- 没有 token 用量统计，成本不可见；
- `logs` 数组会随任务长度增长，长任务里 checkpoint 体积会变大。

## 改进建议

引入 `structlog` + `contextvars` 注入 `thread_id` 与 `node`，把每次模型调用的耗时和 token 数写入一张独立的 SQLite 表，即可获得最小可用的成本与性能视图。
