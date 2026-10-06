from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness.config import settings
from harness.metrics import gating_metrics


def test_mean_faithfulness_meets_gate(golden_cases):
    metric = gating_metrics()["faithfulness"]
    scores = []
    for case in golden_cases:
        metric.measure(case)
        scores.append(metric.score)
    mean = sum(scores) / len(scores)
    print(f"\nmean faithfulness = {mean:.3f} (gate >= {settings.faithfulness_min})")
    assert mean >= settings.faithfulness_min, (
        f"faithfulness {mean:.3f} below gate {settings.faithfulness_min}"
    )
