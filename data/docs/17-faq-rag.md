# 常见问题：RAG 效果不好

## 先做诊断，再改参数

调用 `POST /retrieve` 单独观察检索结果：

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"RRF 怎么融合两路结果","top_k":5}'
```

看返回的 `hits` 里有没有正确文档。如果 top 5 都召回不到，问题在索引或分词；如果能召回但排名靠后，问题在融合或切分。

## 症状一：完全召回不到

1. **索引没建**：`GET /index/stats` 里 `documents` 是 0，去跑 `python tools/ingest.py`；
2. **换了 embedding 后端没重建索引**：维度不一致，必须重建；
3. **语料本身没有相关内容**：这是最常见也最容易被忽略的原因，先确认语料覆盖度。

## 症状二：召回得到但不相关

1. 调小 `CHUNK_SIZE`（如 500 → 300），让每个片段更聚焦；
2. 调大 `CHUNK_OVERLAP` 到 100 左右，减少切割点破坏语义；
3. 调大 `RAG_TOP_K` 到 5 观察是否只是排名问题；
4. 检查 `tokenize` 对中文的处理是否符合语料特征。

## 症状三：资料进了 Prompt 但模型没用

1. Worker 的 Prompt 已要求「优先依据资料作答并标注来源编号」，可以在输出里检查是否有 `[1]` 标注；
2. 片段太长导致被淹没，调小 `CHUNK_SIZE`；
3. `RAG_TOP_K` 过大引入噪声，反而稀释了有效信息。

## 症状四：本地哈希向量语义能力弱

哈希向量只做字面投影，没有真正的语义泛化。如果任务需要「同义不同词」的召回，应改用 `EMBEDDING_PROVIDER=openai` 接一个真正的语义向量服务。这是本框架 RAG 部分最大的已知能力边界。
