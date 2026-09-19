# 文档加载与切分策略

## 加载

`rag/loader.py` 扫描 `DOCS_DIR`（默认 `data/docs`）下的所有 `.md` / `.markdown` / `.txt` 文件，递归读取，按文件名排序保证顺序稳定。如果文件首行是 `# 标题`，就把标题提取出来放进 metadata，方便最终展示引用来源。

## 切分

优先使用 LangChain 的 `RecursiveCharacterTextSplitter`，参数为 `chunk_size=500`、`chunk_overlap=80`。

分隔符优先级为：`\n\n` → `\n` → `。` → `；` → `. ` → 空格 → 空串。这个顺序是「先按语义边界切，切不动再退化为硬切」。

如果环境里没装 `langchain-text-splitters`，会使用内置的等价实现，逻辑一致：先按分隔符递归拆分，再把过短的片段合并，并在相邻 chunk 之间保留 overlap。

## 为什么需要 overlap

相邻 chunk 之间重叠 80 个字符，能避免「关键结论正好落在切割点上」导致的语义断裂。代价是索引体积略微膨胀，对 20~30 篇文档的规模完全可以接受。

## chunk 的 id 与 metadata

每个 chunk 的 id 形如 `04-worker.md#2`，metadata 包含：

- `source`：文件名，Worker 引用时显示为「来源」；
- `title`：文档标题；
- `chunk`：在文档内的序号。

## 切分粒度怎么调

- **chunk 太大**（如 1000）：单个片段信息过载，检索精度下降，Prompt 里挤占上下文；
- **chunk 太小**（如 150）：语义不完整，容易召回一堆没有结论的碎片。

500 字左右对中文技术文档是比较稳的默认值。如果文档是高度结构化的（如 API 文档），可以适当调小到 300。
