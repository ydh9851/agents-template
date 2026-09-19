"""命令行跑一个任务（不启动服务），用于本地验证最小链路。

用法：
    python tools/run_task.py "帮我写一份多 Agent 框架的技术选型说明"
    python tools/run_task.py --thread-id abc123 --resume        # 断点续跑
    python tools/run_task.py "任务内容" --json                  # 输出最终状态 JSON
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from graph.workflow import stream_task  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="本地执行一个多 Agent 任务")
    parser.add_argument("task", nargs="?", default="", help="用户任务描述")
    parser.add_argument("--thread-id", default=None, help="会话 id（用于断点续跑）")
    parser.add_argument("--resume", action="store_true", help="从指定 thread_id 的 checkpoint 继续")
    parser.add_argument("--json", action="store_true", help="最后打印完整状态 JSON")
    args = parser.parse_args()

    if not args.resume and not args.task.strip():
        parser.error("首次执行必须提供 task")

    print(f"模型：{settings.model_name}（Mock 模式：{settings.use_mock}）\n" + "-" * 60)

    final_state: dict = {}
    for event, data in stream_task(task=args.task, thread_id=args.thread_id, resume=args.resume):
        if event == "start":
            print(f"[start] thread_id={data['thread_id']} resume={data['resume']}")
        elif event == "node":
            node = data["node"]
            print(f"\n[node] {node}")
            for log in (data.get("update") or {}).get("logs", [])[-1:]:
                print(f"       {log}")
        elif event == "done":
            final_state = data
        elif event == "error":
            print(f"\n[error] {data['message']}")
            return 2

    print("\n" + "=" * 60)
    print("子任务结果：")
    for item in final_state.get("results", []):
        flag = "通过" if item.get("passed") else "未通过"
        print(f"  - {item.get('id')} {item.get('title')} [{flag}] 尝试 {item.get('attempts')} 次")

    print("\n最终交付：\n" + (final_state.get("final_answer") or "（空）"))
    if args.json:
        print("\n" + json.dumps(final_state, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
