"""通用小工具：日志、模板渲染、从模型输出里稳健地抽取 JSON。"""
import json
import logging
import re
import sys
from string import Template
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def get_logger(name: str = "agents") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


logger = get_logger()


def render(template: str, **kwargs: Any) -> str:
    """用 $var 占位符渲染 Prompt 模板（JSON 里的花括号不受影响）。"""
    return Template(template).safe_substitute(**kwargs)


def extract_json(text: str) -> dict:
    """从模型输出中抽取第一个合法 JSON 对象。

    依次尝试：```json 代码块 -> 整段文本 -> 花括号平衡匹配。
    模型偶尔会多写几句解释，这里做容错，避免整条链路因为格式问题中断。
    """
    if not text or not text.strip():
        raise ValueError("模型返回内容为空，无法解析 JSON")

    candidates: list[str] = []
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1))
    candidates.append(text.strip())

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    # 兜底：从第一个 '{' 开始做括号平衡扫描
    start = text.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break

    raise ValueError(f"无法从模型输出中解析出 JSON：{text[:200]}")


def to_jsonable(obj: Any) -> Any:
    """把任意对象转成可 JSON 序列化的结构（用于 SSE 输出）。"""
    return json.loads(json.dumps(obj, ensure_ascii=False, default=str))
