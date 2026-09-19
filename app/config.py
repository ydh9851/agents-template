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

    # ---------- 编排 ----------
    sqlite_path: str = str(ROOT / "data" / "checkpoint.sqlite")
    max_subtasks: int = 5
    # 单个子任务最多允许被 Checker 打回几次
    max_retry_per_subtask: int = 2

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
