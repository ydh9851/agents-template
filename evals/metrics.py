"""评估指标：分类指标 + 描述统计（纯标准库）。

为什么自己写而不引 scikit-learn：
    评估要能在 CI 里零负担跑起来。为了算一个 F1 去装几十 MB 的科学计算栈不划算，
    而这些公式本身只有几行。
"""
from __future__ import annotations


def accuracy(pairs: list[tuple[bool, bool]]) -> float:
    """(预测, 标注) 的一致率。"""
    if not pairs:
        return 0.0
    return sum(1 for predicted, label in pairs if predicted == label) / len(pairs)


def binary_scores(pairs: list[tuple[bool, bool]]) -> dict:
    """二分类的 precision / recall / f1 与混淆矩阵（positive = True）。

    :param pairs: [(预测是否通过, 标注是否通过)]
    """
    tp = sum(1 for predicted, label in pairs if predicted and label)
    fp = sum(1 for predicted, label in pairs if predicted and not label)
    fn = sum(1 for predicted, label in pairs if not predicted and label)
    tn = sum(1 for predicted, label in pairs if not predicted and not label)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": accuracy(pairs),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], p: float) -> float:
    """线性插值分位数，p 取 0~1（例如 0.5 是中位数、0.95 是 P95）。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = max(0.0, min(1.0, p)) * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return float(ordered[low] * (1 - frac) + ordered[high] * frac)
