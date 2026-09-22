"""Worker 可调用的工具集。

设计原则：这是**白名单工具**，不是「给模型一个 shell」。

    - 工具的名字、参数、边界全部写死在代码里，模型只能从中选
    - 所有文件操作都过 app/workspace.py 的路径校验，越界直接拒绝
    - 工具执行失败**不抛异常中断任务**，而是把错误文本回灌给模型让它自己纠正 ——
      模型写了个越界路径，正确反应是告诉它「不允许」，而不是让整个任务崩掉

工具集刻意保持极简（三个）：能看目录、能读、能写，已经足够支撑「产出交付物」这类任务。
真正的编码类任务需要更完整的工具层（见 README 的路线图）。
"""
from __future__ import annotations

import json

from app import sandbox, workspace
from app.config import settings
from app.observability import get_logger, metrics

logger = get_logger("app.tools")

TOOL_SPECS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出工作区内已有的文件（相对路径与字节数）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "glob 模式，默认 **/* 表示全部文件",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取工作区内某个文本文件的内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "工作区内的相对路径，如 report.md"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "把内容写入工作区内的文件（覆盖同名文件）。"
                "适合产出报告、清单、方案等可直接交付的成品。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "工作区内的相对路径，建议带后缀，如 report.md",
                    },
                    "content": {"type": "string", "description": "文件的完整内容"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "在工作区里执行一个 .py 文件并返回真实输出。"
                "写完代码后用它自检；报错信息会原样返回，照着改就行。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "工作区内的 .py 相对路径，如 solution.py",
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "传给脚本的命令行参数（可选）",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": (
                "在工作区里运行 pytest（默认跑 tests/ 目录）并返回完整测试输出。"
                "这是判断「做完了没有」的唯一可信依据 —— 优先于你自己的判断。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "测试路径，默认 tests"},
                },
            },
        },
    },
]

_SPEC_BY_NAME = {spec["function"]["name"]: spec for spec in TOOL_SPECS}

# 会真的把代码跑起来的工具。单独列出来，是为了让 CODE_EXECUTION_ENABLED
# 能一刀切掉它们 —— 文件读写可以靠路径校验兜住，执行不行，得能整体关掉。
EXEC_TOOLS = {"run_python", "run_tests"}


def enabled_tools() -> list[str]:
    """当前启用的工具名（按配置过滤）。"""
    names = [item.strip() for item in (settings.worker_tools or "").split(",") if item.strip()]
    if not settings.code_execution_enabled:
        names = [name for name in names if name not in EXEC_TOOLS]
    return names


def tools_enabled() -> bool:
    return bool(enabled_tools())


def tool_schemas() -> list[dict]:
    """返回 OpenAI function-calling 格式的工具定义，供 bind_tools 使用。"""
    enabled = set(enabled_tools())
    return [spec for spec in TOOL_SPECS if spec["function"]["name"] in enabled]


def tools_prompt_block() -> str:
    """给 prompt 用的一段工具说明（真实模式主要靠 function calling，这段是补充上下文）。"""
    enabled = enabled_tools()
    if not enabled:
        return ""
    lines = ["【你可以使用的工具】"]
    for name in enabled:
        spec = _SPEC_BY_NAME.get(name)
        if spec:
            lines.append(f"- {name}：{spec['function']['description']}")
    lines.append(
        "需要产出成品文件时，直接调用 write_file；不需要落盘时，把结论写在正文里即可。"
    )
    return "\n".join(lines)


def _run_exec_tool(name: str, args: dict) -> dict:
    """执行类工具的统一出口。

    成功和失败都把**完整输出**带回给模型 —— 尤其是失败：
    Coder 正是靠 stderr 里的 traceback 才能定位问题，
    所以这里把输出放在 error 字段里，而不是只回一句「执行失败」。
    """
    if name == "run_python":
        raw_args = args.get("args") or []
        if not isinstance(raw_args, list):
            raw_args = [str(raw_args)]
        result = sandbox.run_python(
            str(args.get("path", "")), [str(item) for item in raw_args]
        )
    else:
        result = sandbox.run_pytest(str(args.get("target") or "tests"))

    text = result.to_text()
    return {"ok": True, "result": text} if result.ok else {"ok": False, "error": text}


def run_tool(name: str, args: dict | None = None) -> dict:
    """执行工具。永远返回 dict（成功含 result / 失败含 error），不抛异常。"""
    args = args or {}
    if name not in enabled_tools():
        return {"ok": False, "error": f"工具未启用：{name}"}

    try:
        if name == "list_files":
            files = workspace.list_files(str(args.get("pattern") or "**/*"))
            return {"ok": True, "result": files or "（工作区当前为空）"}

        if name == "read_file":
            text = workspace.read_text(str(args.get("path", "")))
            return {"ok": True, "result": text}

        if name == "write_file":
            info = workspace.write_text(str(args.get("path", "")), str(args.get("content", "")))
            return {
                "ok": True,
                "result": f"已写入 {info['path']}（{info['size']} 字节）",
                "artifact": info["path"],
            }

        if name in EXEC_TOOLS:
            return _run_exec_tool(name, args)

    except workspace.WorkspaceError as exc:
        # 越界/后缀不合法这类错误回灌给模型，让它自己改正 —— 这正是 ReAct 循环的价值
        metrics.inc("tool_calls_total", tool=name, status="rejected")
        logger.warning("工具 %s 被拒绝：%s", name, exc)
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - 工具失败不该拖垮整个任务
        metrics.inc("tool_calls_total", tool=name, status="error")
        logger.warning("工具 %s 执行失败：%s", name, exc)
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    metrics.inc("tool_calls_total", tool=name, status="unknown_tool")
    return {"ok": False, "error": f"未知工具：{name}"}


def tool_result_text(result: dict) -> str:
    """把工具结果转成回灌给模型的一段文本。"""
    if result.get("ok"):
        payload = result.get("result")
        if isinstance(payload, (dict, list)):
            payload = json.dumps(payload, ensure_ascii=False)
        return str(payload)
    return f"[工具报错] {result.get('error')}"
