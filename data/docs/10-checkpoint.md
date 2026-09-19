# 断点续跑：SQLite Checkpoint

## 原理

LangGraph 的 checkpointer 会在**每个节点执行完成后**自动把完整状态序列化写入存储。本项目使用官方的 `SqliteSaver`（`langgraph-checkpoint-sqlite`），落盘位置由 `SQLITE_PATH` 指定，默认 `data/checkpoint.sqlite`。

## 为什么不用自己写持久化

因为 LangGraph 已经把「节点边界 = 安全点」这件事定义清楚了。自己写持久化需要在每个节点里手动保存状态，还要处理并发写、字段演进、恢复时的状态校验，成本远高于直接用一个 checkpointer。

## 会话隔离

每个任务用 `thread_id` 标识。配置形如：

```python
{"configurable": {"thread_id": "abc123"}}
```

同一个 `thread_id` 的多次执行共享一条状态链，不同 `thread_id` 互不可见。

## 怎么续跑

1. **进程内中断**：直接再次调用即可。
2. **进程重启后恢复**：使用同一个 `thread_id` 并带 `resume=true` 调用，例如：

```bash
python tools/run_task.py --thread-id abc123 --resume
```

```bash
curl -X POST http://localhost:8000/task \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"abc123","resume":true}'
```

3. **查询状态**：`GET /task/{thread_id}/state` 会返回该 thread 的最新状态和 `next`（即接下来将要执行的节点），可以用来确认 checkpoint 是否真的写进去了。

## 降级说明

如果环境里没有安装 `langgraph-checkpoint-sqlite`，代码会自动降级为 `MemorySaver`：进程内可以续跑，但进程重启后状态丢失，日志里会有明确告警。

## 注意点

SQLite 连接创建时使用了 `check_same_thread=False`，因为 FastAPI 会把同步图执行放到线程池里。同时 `SqliteSaver.setup()` 会在首次使用时自动建表，无需手动迁移。
