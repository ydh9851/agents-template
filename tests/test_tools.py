"""工具层测试：执行、边界、错误回灌。

最关键的一条是 `test_escape_returns_error_not_exception`：
越界必须变成「工具错误」交回模型，而不是抛异常把整个任务打断 ——
这正是 ReAct 循环存在的意义（模型收到错误后通常能自己改对路径）。
"""
import pytest

from app import tools
from app.config import settings


@pytest.fixture
def ws(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "workspace_dir", str(tmp_path / "ws"))
    monkeypatch.setattr(settings, "worker_tools", "list_files,read_file,write_file")
    return tmp_path


def test_write_file_returns_artifact(ws):
    result = tools.run_tool("write_file", {"path": "r.md", "content": "内容"})
    assert result["ok"] is True
    assert result["artifact"] == "r.md"


def test_read_file(ws):
    tools.run_tool("write_file", {"path": "r.md", "content": "内容"})
    assert tools.run_tool("read_file", {"path": "r.md"})["result"] == "内容"


def test_list_files(ws):
    tools.run_tool("write_file", {"path": "a.md", "content": "x"})
    result = tools.run_tool("list_files", {})
    assert result["ok"] is True
    assert [item["path"] for item in result["result"]] == ["a.md"]


def test_list_files_empty_workspace(ws):
    result = tools.run_tool("list_files", {})
    assert result["ok"] is True
    assert "为空" in str(result["result"])


def test_escape_returns_error_not_exception(ws):
    result = tools.run_tool("write_file", {"path": "../evil.md", "content": "x"})
    assert result["ok"] is False
    assert "越界" in result["error"]


def test_disallowed_suffix_returns_error(ws):
    result = tools.run_tool("write_file", {"path": "evil.exe", "content": "x"})
    assert result["ok"] is False
    assert "不支持的文件类型" in result["error"]


def test_disabled_tool_is_rejected(ws, monkeypatch):
    monkeypatch.setattr(settings, "worker_tools", "list_files")
    result = tools.run_tool("write_file", {"path": "a.md", "content": "x"})
    assert result["ok"] is False
    assert "未启用" in result["error"]


def test_unknown_tool_is_rejected(ws):
    result = tools.run_tool("rm_rf", {})
    assert result["ok"] is False


def test_tools_prompt_block_lists_enabled(ws):
    block = tools.tools_prompt_block()
    assert "write_file" in block
    assert "read_file" in block


def test_tools_prompt_block_empty_when_disabled(ws, monkeypatch):
    monkeypatch.setattr(settings, "worker_tools", "")
    assert tools.tools_prompt_block() == ""
    assert tools.tools_enabled() is False


def test_tool_schemas_follow_config(ws, monkeypatch):
    monkeypatch.setattr(settings, "worker_tools", "read_file")
    names = [spec["function"]["name"] for spec in tools.tool_schemas()]
    assert names == ["read_file"]


def test_tool_result_text_formats_both_cases(ws):
    assert tools.tool_result_text({"ok": True, "result": [{"path": "a.md"}]})
    assert tools.tool_result_text({"ok": False, "error": "boom"}).startswith("[工具报错]")


def test_missing_required_arg_is_handled(ws):
    """模型漏传参数时必须返回错误，而不是抛 KeyError 打断任务。"""
    result = tools.run_tool("read_file", {})
    assert result["ok"] is False
