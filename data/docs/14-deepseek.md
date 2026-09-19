# DeepSeek API 接入

## 协议

DeepSeek 提供 OpenAI 兼容的接口，所以直接复用 `langchain-openai` 的 `ChatOpenAI`，只需把 `base_url` 指向 `https://api.deepseek.com`。

```python
ChatOpenAI(
    model="deepseek-chat",
    api_key=settings.deepseek_api_key,
    base_url=settings.deepseek_base_url,
    temperature=0.2,
    timeout=120,
    max_retries=2,
)
```

## 可用模型

- `deepseek-chat`：通用对话模型，速度快、成本低，适合本框架的编排与执行场景；
- `deepseek-reasoner`：推理模型，适合拆解复杂任务，但返回内容里可能包含思维链，解析 JSON 时要注意。

## Mock 降级机制

`settings.use_mock` 的判定逻辑是「显式设置了 `MOCK_LLM=true`」**或**「`DEEPSEEK_API_KEY` 为空」。这保证了两件事：

1. 新克隆项目、没配 Key 也能立刻跑通全链路（输出是占位内容，但流程完整）；
2. CI 环境不需要持有真实 Key 就能做集成测试。

MockLLM 通过 Prompt 首行的 `<!--ROLE=xxx-->` 标记识别当前角色，从而返回结构合法的假 JSON，因此它能真实地驱动状态机流转，包括 Checker 的通过分支。

## 成本控制建议

- 编排类调用统一用低温度，减少无意义的重试；
- `MAX_SUBTASKS` 控制在 5 以内，子任务越多总 token 消耗越大；
- `RAG_TOP_K` 默认 3，调大既增加成本又可能引入噪声；
- 打回重试是成本放大器，把 `acceptance` 写清楚比事后重试更划算。
