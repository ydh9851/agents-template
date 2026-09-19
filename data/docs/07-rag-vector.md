# 向量检索与 Embedding 选择

向量检索是 RAG 的第二条线，负责召回「字面不重合但语义相近」的片段。

## 两种 Embedding 后端

### local（默认）：本地哈希向量

用 hashing trick 把 token 投影到固定维度（默认 384）的向量空间：

1. 对每个 token 做 MD5，前 4 字节取模得到维度下标；
2. 第 5 字节的奇偶决定符号（+1 / -1），这是一种符号哈希，能部分抵消碰撞带来的偏差；
3. 累加后做 L2 归一化。

优点是零依赖、零下载、确定性可复现，非常适合本地开发和 CI；缺点是它本质是**字面匹配的量化版本**，不具备真正的语义泛化能力。

### openai：调用远程 embedding 接口

设置 `EMBEDDING_PROVIDER=openai`，并配置 `EMBEDDING_API_KEY` 与 `EMBEDDING_BASE_URL`。任何 OpenAI 兼容的 `/embeddings` 服务都可以接。

**注意**：DeepSeek 官方目前只提供对话模型，没有 embedding 接口。如果你需要真正的语义向量，需要把 `EMBEDDING_BASE_URL` 指向其它支持 embeddings 的服务（例如本地部署的 BGE 服务或其它云厂商）。

## 向量库

优先使用 ChromaDB 持久化到 `data/chroma`，集合的相似度度量设为 `cosine`。如果环境里没有 chromadb（它依赖较重），会自动降级为把向量存成 JSON 文件的 `SimpleVectorStore`，用 numpy 做余弦相似度矩阵运算。两条路径的接口完全一致，业务代码无感知。

## 维度一致性

切换 embedding 后端后必须重建索引（`python tools/ingest.py`），否则新旧向量维度不一致会导致查询报错。
