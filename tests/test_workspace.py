"""工作区路径安全测试。

这是整个项目里最该「宁可多测」的一块：路径校验一旦有疏漏，
Worker 就能读写工作区之外的任意文件。所有已知的绕过手法都要有用例钉住。
"""
import os

import pytest

from app import workspace
from app.config import settings


@pytest.fixture
def ws(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "workspace_dir", str(tmp_path / "ws"))
    monkeypatch.setattr(settings, "artifact_max_kb", 1)  # 1KB 上限，方便测超限
    return workspace.workspace_root()


# ---------------------------------------------------------------- 正常读写
def test_write_then_read_roundtrip(ws):
    info = workspace.write_text("a/b.md", "hello 世界")
    assert info["path"] == "a/b.md"
    assert workspace.read_text("a/b.md") == "hello 世界"


def test_write_creates_parent_dirs(ws):
    workspace.write_text("deep/nested/dir/note.md", "x")
    assert (ws / "deep" / "nested" / "dir" / "note.md").exists()


def test_list_files(ws):
    workspace.write_text("x.md", "1")
    workspace.write_text("sub/y.txt", "2")
    paths = {item["path"] for item in workspace.list_files()}
    assert paths == {"x.md", "sub/y.txt"}


# ---------------------------------------------------------------- 路径逃逸
def test_rejects_absolute_path(ws, tmp_path):
    with pytest.raises(workspace.WorkspaceError):
        workspace.resolve(str(tmp_path / "outside.md"))


def test_rejects_parent_traversal(ws):
    for bad in ["../secret.md", "../../secret.md", "a/../../secret.md"]:
        with pytest.raises(workspace.WorkspaceError):
            workspace.resolve(bad)


def test_rejects_sibling_dir_with_same_prefix(ws, tmp_path):
    """`ws_evil` 与 `ws` 同前缀。

    如果用字符串 startswith 判断归属，这种目录会被误判为「在工作区内」；
    这里用 Path.relative_to 做真正的目录归属判断，所以必须拦住。
    """
    evil = tmp_path / "ws_evil"
    evil.mkdir()
    (evil / "x.md").write_text("机密", encoding="utf-8")

    with pytest.raises(workspace.WorkspaceError):
        workspace.read_text("../ws_evil/x.md")


def test_rejects_symlink_escape(ws, tmp_path):
    """工作区里放一个指向外部的软链 —— 绝对路径与前缀检查都拦不住它。"""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("机密", encoding="utf-8")

    link = ws / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("当前环境不支持创建符号链接（Windows 需管理员权限）")

    with pytest.raises(workspace.WorkspaceError):
        workspace.read_text("link/secret.md")


# ---------------------------------------------------------------- 内容约束
def test_rejects_disallowed_suffix(ws):
    for bad in ["evil.exe", "script.sh", "no_suffix"]:
        with pytest.raises(workspace.WorkspaceError):
            workspace.write_text(bad, "x")


def test_rejects_oversized_content(ws):
    with pytest.raises(workspace.WorkspaceError):
        workspace.write_text("big.md", "x" * (2 * 1024))  # 上限 1KB


def test_read_missing_file(ws):
    with pytest.raises(workspace.WorkspaceError):
        workspace.read_text("nope.md")


def test_read_directory_is_rejected(ws):
    workspace.write_text("sub/a.md", "x")
    with pytest.raises(workspace.WorkspaceError):
        workspace.read_text("sub")


def test_empty_path_is_rejected(ws):
    with pytest.raises(workspace.WorkspaceError):
        workspace.resolve("   ")


def test_paths_are_normalized_to_forward_slash(ws):
    """Windows 上反斜杠会让 prompt 里拼出来的路径很难看，统一成正斜杠。"""
    info = workspace.write_text("a\\b.md", "x")
    assert "\\" not in info["path"]
    assert info["path"] == "a/b.md"


def test_workspace_error_is_a_plain_exception():
    """它会被 tools.run_tool 捕获后回灌给模型，所以不能继承 BaseException。"""
    assert issubclass(workspace.WorkspaceError, Exception)
    assert not issubclass(workspace.WorkspaceError, BaseException.__class__)


def test_root_is_created_on_demand(monkeypatch, tmp_path):
    target = tmp_path / "not" / "yet" / "created"
    monkeypatch.setattr(settings, "workspace_dir", str(target))
    root = workspace.workspace_root()
    assert root.exists()
    assert os.path.samefile(root, target)
