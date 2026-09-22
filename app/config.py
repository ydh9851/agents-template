"""全局配置层。

所有配置项统一从 `.env` 读取（pydantic-settings），代码里不出现任何硬编码密钥。
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（本文件位于 <root>/app/config.py）
ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """应用配置。字段名不区分大小写，环境变量同名即可覆盖。"""

    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- LLM（DeepSeek，OpenAI 兼容协议） ----------
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    model_name: str = "deepseek-chat"
    temperature: float = 0.2
    request_timeout: int = 120
    # 强制使用离线 Mock 模型（不配置时：没 Key 就自动降级为 Mock）
    mock_llm: bool = False
    # Mock 模式下让 Checker 前 N 次判定不通过，用于离线验证「打回重做」分支
    mock_checker_fail_times: int = 0

    # ---------- RAG ----------
    docs_dir: str = str(ROOT / "data" / "docs")
    index_dir: str = str(ROOT / "data" / "index")
    chroma_dir: str = str(ROOT / "data" / "chroma")
    chunk_size: int = 500
    chunk_overlap: int = 80
    rag_top_k: int = 3
    # RRF 融合公式里的常数 k
    rrf_k: int = 60

    # Embedding：local = 零依赖本地哈希向量；openai = 任意 OpenAI 兼容的 embedding 服务
    embedding_provider: str = "local"
    embedding_dim: int = 384
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    # 单次 embedding 请求最多提交多少条文本
    embedding_batch_size: int = 64

    # ---------- Rerank（RRF 融合之后的精排） ----------
    # none          不重排，直接用融合顺序（基线）
    # heuristic     零依赖启发式：融合分 + 查询词覆盖 + 标题命中（默认）
    # cross_encoder 交叉编码器，精度最高（需 sentence-transformers + 下载模型）
    # llm           让现有 LLM 给候选打分（无需额外模型，但每次检索多一次 API 调用）
    rerank_provider: str = "heuristic"
    # 送入重排的候选数：先从融合结果里取这么多，再精排到 rag_top_k。
    # 池子太小就没什么可排的，太大则重排成本上升（cross_encoder 尤其明显）
    rerank_pool: int = 10
    rerank_model: str = "BAAI/bge-reranker-base"

    # ---------- LLM 可靠性 ----------
    # 单次调用失败后的重试次数（不含首次）
    llm_max_retries: int = 2
    # 重试退避基数（秒），第 n 次重试等待 base * 2^(n-1)
    llm_retry_backoff: float = 0.8
    # 整个调用（含所有重试）的总时间预算（秒），防止一次请求把节点占死
    llm_total_timeout: int = 120
    # 主模型失败后的备用模型，逗号分隔（如 deepseek-reasoner）
    fallback_models: str = ""
    # 单个任务的 token 预算，超出后不再调用真实模型（0 = 不限制）
    task_token_budget: int = 0

    # ---------- 可观测性 ----------
    # text = 人读友好（默认）；json = 一行一个 JSON，便于采集
    log_format: str = "text"

    # ---------- 安全 ----------
    # 逗号分隔的 API Key；留空表示不开启鉴权（保持开箱即用）
    api_keys: str = ""
    # 每个客户端的每分钟请求上限（0 = 不限制）
    rate_limit_per_minute: int = 0
    # 限流后端：memory = 单进程内存（默认，零依赖）；redis = 多副本共享计数
    rate_limit_backend: str = "memory"
    # Redis 连接串（仅 rate_limit_backend=redis 时使用；redis 包可选装）
    redis_url: str = "redis://localhost:6379/0"
    # 是否信任反向代理传来的 X-Forwarded-For（本地直连时不要开）
    trust_proxy: bool = False

    # ---------- 工具与工作区 ----------
    # Worker 能调用的工具（逗号分隔）；留空表示不开放，退化为纯文本产出。
    # 开放文件/执行权限前请先读 app/workspace.py 与 app/sandbox.py —— 安全边界在那两个文件里。
    worker_tools: str = "list_files,read_file,write_file,run_python,run_tests"
    # 单个子任务最多几轮工具调用（每轮 = 一次模型调用 + 若干工具执行），防止无限循环烧 token
    max_tool_rounds: int = 3
    # 单个产物文件大小上限（KB）
    artifact_max_kb: int = 256
    # 工作区目录：所有文件读写都被限制在这个目录内
    workspace_dir: str = str(ROOT / "data" / "workspace")

    # ---------- 代码沙箱 ----------
    # subprocess = 子进程执行（默认，零依赖，但**不是隔离**：同用户、能读文件系统）
    # docker     = 容器隔离（--network=none + 内存/CPU/PID 上限），推荐用于不可信代码
    sandbox_backend: str = "subprocess"
    # 单次执行超时（秒）。超时会杀掉整个进程组，不只是主进程
    sandbox_timeout: int = 20
    # docker 后端的内存上限（MB）与镜像
    sandbox_max_memory_mb: int = 256
    sandbox_docker_image: str = "python:3.11-slim"
    # 子进程输出保留的最大字符数（超出截断，防止一行 print 刷爆内存与上下文）
    sandbox_output_limit: int = 4000

    # ---------- 编码任务（Coder / Tester）----------
    # 关掉后 Worker 不再执行代码，退化为「写文件 + 文本产出」
    code_execution_enabled: bool = True
    # Coder 的工具轮次上限（一轮 = 一次模型调用 + 若干工具执行）。
    # 写实现 → 跑测试 → 看报错 → 修，通常 3~4 轮够；给到 6 是为了留出反复调试的余地
    coding_max_rounds: int = 6
    # Tester 独立写测试的轮次上限（它只需要写文件，2 轮足够）
    testing_max_rounds: int = 2

    # ---------- 编排 ----------
    sqlite_path: str = str(ROOT / "data" / "checkpoint.sqlite")
    max_subtasks: int = 5
    # 单个子任务最多允许被 Checker 打回几次
    max_retry_per_subtask: int = 2
    # 同一层就绪子任务的并行执行上限（LLM 调用是 IO 密集型，并发能显著缩短总耗时）
    max_parallel_subtasks: int = 3

    # ---------- 服务 ----------
    host: str = "0.0.0.0"
    port: int = 8000
    # 逗号分隔；默认 * 便于本地联调（此时不允许携带 Cookie）
    cors_origins: str = "*"

    # ---------- 派生属性 ----------
    @property
    def use_mock(self) -> bool:
        """没有 API Key 时自动走 Mock，保证最小链路始终可跑通。"""
        return bool(self.mock_llm) or not self.deepseek_api_key.strip()

    @property
    def cors_origin_list(self) -> list[str]:
        items = [x.strip() for x in self.cors_origins.split(",") if x.strip()]
        return items or ["*"]

    @property
    def api_key_set(self) -> set[str]:
        """开启鉴权所需的合法 Key 集合；为空表示不鉴权。"""
        return {x.strip() for x in self.api_keys.split(",") if x.strip()}

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key_set)

    @property
    def fallback_model_list(self) -> list[str]:
        """主模型 + 备用模型，去重保序。"""
        chain = [self.model_name]
        for item in self.fallback_models.split(","):
            name = item.strip()
            if name and name not in chain:
                chain.append(name)
        return chain

    def ensure_dirs(self) -> None:
        """确保运行时目录存在。"""
        for path in (self.docs_dir, self.index_dir, self.chroma_dir):
            Path(path).mkdir(parents=True, exist_ok=True)
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


settings = get_settings()
