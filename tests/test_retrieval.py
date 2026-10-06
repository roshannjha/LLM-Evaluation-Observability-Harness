from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness.metrics import soft_metrics


def test_report_retrieval_quality(golden_cases):
    metrics = soft_metrics()
    recall = metrics["contextual_recall"]
    precision = metrics["contextual_precision"]

    def mean_over(metric):
        vals = []
        for case in golden_cases:
            metric.measure(case)
            vals.append(metric.score)
        return sum(vals) / len(vals)

    r = mean_over(recall)
    p = mean_over(precision)
    print(f"\nmean context recall = {r:.3f} | mean context precision = {p:.3f}")
    assert 0.0 <= r <= 1.0 and 0.0 <= p <= 1.0
