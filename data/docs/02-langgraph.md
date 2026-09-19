# LangGraph 状态机编排

框架使用 LangGraph 0.2.x 的 `StateGraph` 描述多 Agent 协作流程。整个系统只有一张图，四个节点。

## 节点与边

```
manager ──> worker ──> checker ──┬──(未通过)──> worker
                                 └──(通过/超限)──> summarize ──> END
```

- `manager`：入口节点，读取用户任务，输出子任务列表。
- `worker`：按 `current_index` 取出当前子任务并执行。
- `checker`：验收当前子任务的产出，并决定是否推进下标。
- `summarize`：出口节点，把所有结果汇总成最终交付物。

## 条件边

`route_after_checker` 是关键路由函数：如果 `current_index < len(subtasks)` 就回到 `worker`，否则去 `summarize`。这样「重试」和「继续下一个」都用同一套逻辑表达，不需要额外的循环控制节点。

## 状态合并

LangGraph 默认对节点返回的字段做覆盖合并。因为本项目没有为 list 字段配置 reducer，所以每个节点都必须返回**完整的新列表**，例如 `append_log` 会返回 `state["logs"] + [新日志]`，否则历史日志会被覆盖掉。

## 为什么选 LangGraph

相比自己写 while 循环，LangGraph 带来两个直接收益：一是 checkpointer 让「每个节点自动落盘」零成本；二是 `stream_mode="updates"` 天然支持把节点级进度推给前端。
