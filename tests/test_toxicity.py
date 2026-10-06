from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness.config import settings
from harness.metrics import gating_metrics


def test_mean_toxicity_within_gate(golden_cases):
    metric = gating_metrics()["toxicity"]
    safeties = []
    for case in golden_cases:
        metric.measure(case)
        safeties.append(metric.score)
    mean_toxicity = 1.0 - (sum(safeties) / len(safeties))
    print(f"\nmean toxicity = {mean_toxicity:.3f} (gate <= {settings.toxicity_max})")
    assert mean_toxicity <= settings.toxicity_max, (
        f"toxicity {mean_toxicity:.3f} exceeds gate {settings.toxicity_max}"
    )
