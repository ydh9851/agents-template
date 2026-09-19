# SSE 流式输出

框架通过 Server-Sent Events 把每个节点的执行进度实时推给调用方，接口是 `POST /task/stream`。

## 事件类型

| 事件名 | 含义 | data 字段 |
| --- | --- | --- |
| `start` | 任务开始 | `thread_id`、`resume`、`task` |
| `node` | 某个节点执行完成 | `node`（节点名）、`update`（该节点的状态增量） |
| `done` | 全部结束 | 完整最终状态（含 `results`、`final_answer`、`status`） |
| `error` | 执行异常 | `message` |

节点级事件让前端可以画出实时的执行流水：Manager 拆出几个子任务、Worker 正在做第几个、Checker 打回了哪一条。

## 实现方式

优先使用 `sse-starlette` 的 `EventSourceResponse`；如果环境里没装它，会降级为手写的标准 SSE 帧：

```
event: node
data: {"node":"worker","update":{...}}

```

注意 SSE 要求每条消息以两个换行结尾，并且响应头要带 `Cache-Control: no-cache`、关闭 Nginx 缓冲（`X-Accel-Buffering: no`）。

## 为什么用 POST 而不是 GET

因为任务描述可能很长，放 URL 里会超出长度限制。SSE 本身只是「服务端单向长连接」，规范上没有禁止 POST，用 `fetch` + `ReadableStream` 或 `curl -N` 都能消费。

## 消费示例

```bash
curl -N -X POST http://localhost:8000/task/stream \
  -H "Content-Type: application/json" \
  -d '{"task":"帮我整理一份多 Agent 框架的技术选型说明"}'
```

## 已知限制

- 目前**不支持断线续传**（Last-Event-ID）。连接断开后需要重新发起并配合 `resume=true` 从 checkpoint 继续，而不是从断点事件继续。
- 没有心跳机制，长时间无节点输出时某些反向代理会主动断开连接。
