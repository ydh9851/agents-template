"""离线评估：Manager 拆解质量 / Checker 判定准确率 / 端到端任务完成情况。

为什么在 pytest 之外还要一套评估：
    pytest 检查的是「功能对不对」—— 函数返回值、状态流转、重试分支是否按预期走。
    评估回答的是「输出好不好」—— 拆解有没有覆盖关键点、验收判定跟人的判断是否一致、
    端到端跑一个任务要多久、烧多少 token、重试几次。
    前者保证改动不退化，后者才能说明这套 Agent 协作到底有没有效果。

    （Checker 评估需要真实模型才有意义：Mock 下它恒定放行，
      所以脚本会在 Mock 模式自动跳过该项的阈值判定，而不是给一个虚假的及格分。）

用法：
    python evals/run_eval.py                       # 跑全部
    python evals/run_eval.py --suite manager       # 只跑拆解评估
    python evals/run_eval.py --suite manager,e2e
    python evals/run_eval.py --min-manager 0.9
退出码非 0 表示未达标（CI 据此判定）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows 控制台默认 GBK，直接打印中文或 ✓ 之类的符号会抛 UnicodeEncodeError，
# 把评估脚本自身搞崩。统一成 UTF-8，并且下面只用纯 ASCII 标记兜底。
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001 - 不支持 reconfigure 的环境忽略即可
    pass

from app.config import settings  # noqa: E402
from evals.metrics import binary_scores, mean, percentile  # noqa: E402

CASES_DIR = Path(__file__).resolve().parent / "cases"
SUITES = ("manager", "checker", "retrieval", "e2e")


def _load(name: str) -> dict:
    path = CASES_DIR / name
    if not path.exists():
        raise SystemExit(f"[FAIL] 找不到用例文件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Manager：拆解质量（规则打分）
# ---------------------------------------------------------------------------
def _manager_checks(case: dict, subtasks: list[dict]) -> dict[str, bool]:
    texts = " ".join(
        f"{s.get('title', '')} {s.get('description', '')} {s.get('acceptance', '')}"
        for s in subtasks
    )
    ids = [str(s.get("id")) for s in subtasks]

    deps_ok = True
    for index, sub in enumerate(subtasks):
        for dep in sub.get("deps") or []:
            # 依赖只能指向「已经出现过」的子任务，否则图没法按序执行
            if str(dep) not in ids[:index]:
                deps_ok = False

    return {
        "数量在预期区间": case["min_subtasks"] <= len(subtasks) <= case["max_subtasks"],
        "每个子任务都有验收标准": all(str(s.get("acceptance") or "").strip() for s in subtasks),
        "每个子任务都有描述": all(str(s.get("description") or "").strip() for s in subtasks),
        "依赖只指向已出现的子任务": deps_ok,
        "覆盖了关键点": all(k in texts for k in case.get("must_cover", [])),
        "没有出现被禁的拆解方式": not any(p in texts for p in case.get("forbid_patterns", [])),
    }


def eval_manager() -> dict:
    from agents.manager import split_task

    data = _load("manager_cases.json")
    rows = []
    for case in data["cases"]:
        subtasks = split_task(case["task"])
        checks = _manager_checks(case, subtasks)
        rows.append(
            {
                "id": case["id"],
                "task": case["task"],
                "score": sum(1 for ok in checks.values() if ok) / len(checks),
                "checks": checks,
                "subtask_count": len(subtasks),
            }
        )
    return {"rows": rows, "score": mean([r["score"] for r in rows])}


def print_manager(result: dict) -> None:
    print("\n" + "=" * 74)
    print("Manager 拆解质量（规则打分：结构合法性 + 覆盖度 + 禁用模式）")
    print("=" * 74)
    for row in result["rows"]:
        print(f"  [{row['id']}] {row['task']}")
        print(f"        子任务 {row['subtask_count']} 个，得分 {row['score']:.2f}")
        for name, ok in row["checks"].items():
            if not ok:
                print(f"        [NG] {name}")
    print("-" * 74)
    print(f"平均得分：{result['score']:.3f}")


# ---------------------------------------------------------------------------
# 2. Checker：判定准确率
# ---------------------------------------------------------------------------
def eval_checker() -> dict:
    from agents.checker import build_checker_prompt
    from app.llm import chat
    from app.prompts import load_prompt
    from app.utils import extract_json

    data = _load("checker_cases.json")
    pairs: list[tuple[bool, bool]] = []
    rows = []

    for case in data["cases"]:
        expected = bool(case["expect_pass"])
        prompt = build_checker_prompt(case["subtask"], case["output"], case.get("attempts", 1))
        try:
            verdict = extract_json(chat(load_prompt("checker"), prompt))
            predicted = bool(verdict.get("pass"))
            reason = str(verdict.get("reason", ""))
        except Exception as exc:  # noqa: BLE001 - 与节点一致：解析失败按通过放行
            predicted, reason = True, f"判定异常，按通过放行：{exc}"

        pairs.append((predicted, expected))
        rows.append(
            {
                "id": case["id"],
                "expected": expected,
                "predicted": predicted,
                "reason": reason[:70],
                "ok": predicted == expected,
            }
        )

    return {"rows": rows, "scores": binary_scores(pairs), "total": len(pairs)}


def print_checker(result: dict) -> None:
    scores = result["scores"]
    print("\n" + "=" * 74)
    print("Checker 验收判定（对比人工标注）")
    print("=" * 74)
    for row in result["rows"]:
        mark = "OK" if row["ok"] else "NG"
        print(f"  [{mark}] [{row['id']}] 标注={row['expected']} 判定={row['predicted']}")
        if not row["ok"]:
            print(f"        理由：{row['reason']}")
    print("-" * 74)
    print(f"准确率 {scores['accuracy']:.3f}   Precision {scores['precision']:.3f}   "
          f"Recall {scores['recall']:.3f}   F1 {scores['f1']:.3f}")
    print(f"混淆矩阵：TP={scores['tp']} FP={scores['fp']} FN={scores['fn']} TN={scores['tn']}")


# ---------------------------------------------------------------------------
# 3. 检索质量：Hit@1 / Hit@3 / MRR
# ---------------------------------------------------------------------------
def eval_retrieval() -> dict:
    """用人工标注的口语化提问评测混合检索。

    标注到「文档级」而不是 chunk 级：调 chunk_size 会让 chunk id 变化，
    标注如果绑死到 chunk，改一次切分参数整份标注集就失效了。
    """
    from rag.hybrid_search import get_retriever

    data = _load("retrieval_cases.json")
    k_max = int(data.get("k_max", 5))
    retriever = get_retriever()

    hits1 = hits3 = hitsk = 0
    mrr = 0.0
    rows = []

    for case in data["cases"]:
        expected = set(case["expected"])
        results = retriever.search(case["query"], top_k=k_max)
        sources = [
            (hit.get("metadata") or {}).get("source") or hit.get("source", "")
            for hit in results
        ]
        rank = next((i for i, src in enumerate(sources, start=1) if src in expected), -1)

        if rank == 1:
            hits1 += 1
        if 1 <= rank <= 3:
            hits3 += 1
        if 1 <= rank <= k_max:
            hitsk += 1
            mrr += 1.0 / rank

        rows.append(
            {
                "id": case["id"],
                "query": case["query"],
                "expected": "/".join(sorted(expected)),
                "rank": rank,
                "top1": sources[0] if sources else "(无结果)",
            }
        )

    total = len(rows)
    return {
        "rows": rows,
        "k_max": k_max,
        "hit1": hits1 / total if total else 0.0,
        "hit3": hits3 / total if total else 0.0,
        f"hit{k_max}": hitsk / total if total else 0.0,
        "mrr": mrr / total if total else 0.0,
        "indexed": retriever.stats().get("documents", 0),
    }


def print_retrieval(result: dict) -> None:
    k_max = result["k_max"]
    print("\n" + "=" * 74)
    print(f"混合检索质量（标注 {len(result['rows'])} 条，索引 {result['indexed']} 个 chunk）")
    print("=" * 74)
    for row in result["rows"]:
        if row["rank"] == -1:
            print(f"  [NG] [{row['id']}] {row['query']}")
            print(f"        期望 {row['expected']}，实际首位命中「{row['top1']}」")
    if all(row["rank"] != -1 for row in result["rows"]):
        print(f"  Hit@{k_max}：全部命中")
    print("-" * 74)
    print(f"Hit@1 {result['hit1']:.1%}   Hit@3 {result['hit3']:.1%}   "
          f"Hit@{k_max} {result[f'hit{k_max}']:.1%}   MRR {result['mrr']:.3f}")


# ---------------------------------------------------------------------------
# 4. 端到端：完整链路
# ---------------------------------------------------------------------------
def eval_e2e() -> dict:
    from graph.workflow import run_task

    data = _load("e2e_tasks.json")
    rows = []
    for index, task in enumerate(data["tasks"], start=1):
        started = time.perf_counter()
        state = run_task(task=task, thread_id=f"eval-{index}")
        elapsed = time.perf_counter() - started
        results = state.get("results") or []
        rows.append(
            {
                "task": task,
                "status": state.get("status"),
                "subtasks": len(state.get("subtasks") or []),
                "passed": sum(1 for r in results if r.get("passed")),
                "retries": sum(int(v) for v in (state.get("retries") or {}).values()),
                "tokens": int(state.get("token_usage") or 0),
                "elapsed": elapsed,
                "answered": bool(str(state.get("final_answer") or "").strip()),
            }
        )

    finished = sum(1 for r in rows if r["status"] == "finished" and r["answered"])
    elapsed_list = [r["elapsed"] for r in rows]
    return {
        "rows": rows,
        "finish_rate": finished / len(rows) if rows else 0.0,
        "avg_elapsed": mean(elapsed_list),
        "p95_elapsed": percentile(elapsed_list, 0.95),
        "avg_tokens": mean([float(r["tokens"]) for r in rows]),
        "avg_retries": mean([float(r["retries"]) for r in rows]),
    }


def print_e2e(result: dict) -> None:
    print("\n" + "=" * 74)
    print("端到端任务（完整链路：拆解 → 执行 → 验收 → 汇总）")
    print("=" * 74)
    for row in result["rows"]:
        print(f"  · {row['task']}")
        print(f"      status={row['status']}  子任务={row['subtasks']}  "
              f"通过={row['passed']}  重试={row['retries']}  "
              f"tokens={row['tokens']}  耗时={row['elapsed']:.2f}s")
    print("-" * 74)
    print(f"完成率 {result['finish_rate']:.1%}   平均耗时 {result['avg_elapsed']:.2f}s   "
          f"P95 耗时 {result['p95_elapsed']:.2f}s")
    print(f"平均 tokens {result['avg_tokens']:.0f}   平均重试次数 {result['avg_retries']:.2f}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="all",
                        help="manager / checker / e2e / all，可用逗号组合（如 manager,e2e）")
    parser.add_argument("--min-manager", type=float, default=0.8, help="Manager 平均分下限")
    parser.add_argument("--min-checker", type=float, default=0.7, help="Checker 准确率下限（仅真实模型下判定）")
    parser.add_argument("--min-hit3", type=float, default=0.75,
                        help="检索 Hit@3 下限（索引为空时跳过）。默认值按「本地哈希向量 + 启发式重排」"
                             "的实测水平设定（0.80），留了一点余量；换成真实 embedding 后可以调高")
    parser.add_argument("--min-finish-rate", type=float, default=0.8, help="端到端完成率下限")
    args = parser.parse_args()

    if args.suite == "all":
        suites = list(SUITES)
    else:
        suites = [s.strip() for s in args.suite.split(",") if s.strip()]
        unknown = [s for s in suites if s not in SUITES]
        if unknown:
            print(f"[FAIL] 未知的评估套件：{unknown}，可选 {list(SUITES)}")
            return 1

    mock = settings.use_mock
    print("=" * 74)
    print(f"评估模式：{'Mock（离线）' if mock else f'真实模型 {settings.model_name}'}"
          f"    套件：{', '.join(suites)}")

    failures: list[str] = []

    if "manager" in suites:
        manager = eval_manager()
        print_manager(manager)
        if manager["score"] < args.min_manager:
            failures.append(f"Manager 平均分 {manager['score']:.3f} < 阈值 {args.min_manager:.3f}")

    if "checker" in suites:
        checker = eval_checker()
        print_checker(checker)
        if mock:
            print("  [跳过阈值] Mock 模式下 Checker 恒定放行，准确率不代表真实能力；"
                  "配置 DEEPSEEK_API_KEY 后此项才有意义")
        elif checker["scores"]["accuracy"] < args.min_checker:
            failures.append(
                f"Checker 准确率 {checker['scores']['accuracy']:.3f} < 阈值 {args.min_checker:.3f}"
            )

    if "retrieval" in suites:
        retrieval = eval_retrieval()
        print_retrieval(retrieval)
        if retrieval["indexed"] == 0:
            # 没建索引时检索必然全军覆没，这时判失败没有意义，只提示怎么修
            print("  [跳过阈值] 索引为空，请先执行：python tools/ingest.py")
        elif retrieval["hit3"] < args.min_hit3:
            failures.append(
                f"检索 Hit@3 {retrieval['hit3']:.1%} < 阈值 {args.min_hit3:.1%}"
            )

    if "e2e" in suites:
        e2e = eval_e2e()
        print_e2e(e2e)
        if e2e["finish_rate"] < args.min_finish_rate:
            failures.append(
                f"端到端完成率 {e2e['finish_rate']:.1%} < 阈值 {args.min_finish_rate:.1%}"
            )

    print("\n" + "=" * 74)
    if failures:
        print("[FAIL] " + "；".join(failures))
        return 1
    print("[PASS] 全部评估达标")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
