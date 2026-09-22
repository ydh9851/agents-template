"""受控工作区：把「Worker 能读写文件」这件事关进笼子里。

Worker 原本只能产出文本。一旦开放文件权限，**路径安全就是唯一重要的事** ——
所以所有入口都必须先过 `resolve()`，它负责四道关：

    1. 拒绝绝对路径        否则 /etc/passwd 这种一步就走出去了
    2. 拒绝 .. 越界        ../../ 同样能出去
    3. resolve() 后再校验归属
       —— 这一步专治符号链接：在工作区里放一个指向 / 的软链，前两步都拦不住
    4. 限制后缀与文件大小

宁可报错，也不要留下「看起来能跑、实际能写到任意位置」的口子。
"""
from __future__ import annotations

from pathlib import Path

from app.config import settings
from app.observability import get_logger

logger = get_logger("app.workspace")

# 允许写入的后缀。除 .py 外全是文本类：既贴合本框架「产出可交付物」的定位，
# 也避免写入二进制/可执行文件。
ALLOWED_SUFFIXES = {
    ".md", ".txt", ".json", ".jsonl", ".csv", ".tsv",
    ".yaml", ".yml", ".html", ".xml", ".log",
    # 代码类产物只开放 .py —— 它是沙箱会执行的那一类。
    # 这是刻意权衡的结果：代码任务必须能写出可运行的文件，
    # 而「能被执行」这件事由 app/sandbox.py 里那几条硬性约束兜住，不是靠后缀去拦。
    # 其他可执行类型（.sh / .bat / .exe / .so / .dll）一律不开放。
    ".py",
}

# 列目录时最多返回多少条（防止工作区被塞满后一次吐出海量结果）
MAX_LIST_ITEMS = 200


class WorkspaceError(Exception):
    """路径越界、后缀不允许、超出大小限制等。

    调用方（tools.run_tool）应当把它转成「工具错误」回灌给模型，
    而不是抛给用户 —— 模型收到错误后通常能自己改正路径。
    """


def workspace_root() -> Path:
    root = Path(settings.workspace_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve(relative: str) -> Path:
    """把相对路径解析成工作区内的绝对路径；任何越界都直接拒绝。"""
    raw = (relative or "").strip()
    if not raw:
        raise WorkspaceError("路径不能为空")

    candidate = Path(raw)
    if candidate.is_absolute():
        raise WorkspaceError(f"不允许绝对路径：{raw}")

    root = workspace_root()
    target = (root / candidate).resolve()

    # 关键一步：resolve 之后再判断归属。
    # 用字符串前缀（str.startswith）会被 "workspace_evil/" 这类同前缀目录绕过，
    # 也拦不住符号链接；Path.relative_to 做的是真正的目录归属判断。
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise WorkspaceError(f"路径越界（只能访问工作区内的文件）：{raw}") from exc

    return target


def _ensure_suffix(path: Path) -> None:
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_SUFFIXES))
        raise WorkspaceError(f"不支持的文件类型「{path.suffix or '无后缀'}」；允许：{allowed}")


def _relative(path: Path) -> str:
    # 统一成正斜杠：Windows 上反斜杠会让模型拼出的路径很难看，也容易和 prompt 里的示例不一致
    return str(path.relative_to(workspace_root())).replace("\\", "/")


def list_files(pattern: str = "**/*") -> list[dict]:
    """列出工作区内的文件（相对路径 + 字节数）。"""
    root = workspace_root()
    items: list[dict] = []
    try:
        paths = sorted(root.glob(pattern or "**/*"))
    except (ValueError, OSError) as exc:
        raise WorkspaceError(f"非法的 glob 模式：{pattern}") from exc

    for path in paths:
        if not path.is_file():
            continue
        try:
            items.append({"path": _relative(path), "size": path.stat().st_size})
        except (OSError, ValueError):
            continue
    return items[:MAX_LIST_ITEMS]


def read_text(relative: str, max_bytes: int | None = None) -> str:
    """读取工作区内的文本文件。"""
    path = resolve(relative)
    _ensure_suffix(path)
    if not path.exists():
        raise WorkspaceError(f"文件不存在：{relative}")
    if not path.is_file():
        raise WorkspaceError(f"不是文件：{relative}")

    limit = max_bytes or int(settings.artifact_max_kb) * 1024
    try:
        data = path.read_bytes()[:limit]
    except OSError as exc:
        raise WorkspaceError(f"读取失败：{exc}") from exc
    return data.decode("utf-8", errors="replace")


def write_text(relative: str, content: str) -> dict:
    """写入工作区内的文本文件（会覆盖同名文件）。"""
    path = resolve(relative)
    _ensure_suffix(path)

    encoded = (content or "").encode("utf-8")
    limit = int(settings.artifact_max_kb) * 1024
    if len(encoded) > limit:
        raise WorkspaceError(
            f"内容超出上限 {settings.artifact_max_kb}KB（实际约 {len(encoded) // 1024}KB）"
        )

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
    except OSError as exc:
        raise WorkspaceError(f"写入失败：{exc}") from exc

    logger.info("产物已写入：%s（%d 字节）", _relative(path), len(encoded))
    return {"path": _relative(path), "size": len(encoded)}
