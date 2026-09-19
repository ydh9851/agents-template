# 常见问题：启动与运行

## 启动时报 ModuleNotFoundError: No module named 'xxx'

按顺序排查：

1. 虚拟环境是否激活（`which python` / `where python` 指向 `.venv`）；
2. 依赖是否装全：`pip install -r requirements.txt`；
3. 如果只是缺 `chromadb` 或 `langchain-text-splitters`，其实不影响启动——代码有降级路径，日志里会给出告警。

## 运行 tools/ingest.py 报 ModuleNotFoundError: No module named 'app'

必须以**项目根目录**为工作目录执行：

```bash
cd agents-template
python tools/ingest.py
```

脚本内部已经把项目根目录插进了 `sys.path`，但如果用了 `python -m tools.ingest` 之类的其它方式，仍需保证工作目录正确。

## 接口返回 400：没有加载到任何文档

说明 `DOCS_DIR` 下没读到 markdown。检查目录是否存在、文件后缀是否是 `.md` / `.txt`，以及 `.env` 里的 `DOCS_DIR` 是不是被改成了绝对路径但指向了别处。

## 任务跑完 final_answer 是占位内容

说明当前处于 Mock 模式（`/health` 里 `mock_mode: true`）。配置 `DEEPSEEK_API_KEY` 后重启即可。

## SSE 接口一直没有输出

- 确认用的是 `curl -N`，否则 curl 会缓冲输出；
- 确认反向代理关闭了缓冲（Nginx 需要 `proxy_buffering off;`）；
- 确认任务本身在跑（看服务端日志里的节点执行记录）。

## 端口被占用

```bash
uvicorn main:app --port 8001
```

## Windows 下中文乱码

PowerShell 里先执行 `chcp 65001`。项目所有文件读写都显式指定了 `encoding="utf-8"`，问题一般出在终端而非代码。
