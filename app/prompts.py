"""Prompt 模板加载（带缓存）。

模板放在 prompts/*.txt，用 $var 占位；文件首行 `<!--ROLE=xxx-->` 用于标识角色，
真实调用时会被模型忽略，Mock 模式下则据此返回对应结构的数据。
"""
from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    """读取指定 Prompt 模板，例如 load_prompt('manager')。"""
    path = PROMPT_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt 模板不存在：{path}")
    return path.read_text(encoding="utf-8").strip()
