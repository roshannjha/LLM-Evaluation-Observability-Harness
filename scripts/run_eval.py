from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness.config import REPORTS_DIR, settings
from harness.golden import build_test_case, load_golden
from harness.metrics import gating_metrics, soft_metrics


def _mean(metric, cases) -> float:
    vals = []
    for c in cases:
        try:
            metric.measure(c)
            if metric.score is not None:
                vals.append(metric.score)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! metric error on a case (skipped): {str(exc)[:80]}")
    return sum(vals) / len(vals) if vals else 0.0


def main() -> int:
    triplets = load_golden()
    print(f"Evaluating {len(triplets)} golden triplets with judge "
          f"'{settings.judge_model}' ({settings.llm_backend})...")
    cases = [build_test_case(t) for t in triplets]

    gate = gating_metrics()
    soft = soft_metrics()
    results = {name: _mean(m, cases) for name, m in {**gate, **soft}.items()}

    results["toxicity"] = 1.0 - results["toxicity"]

    faith_ok = results["faithfulness"] >= settings.faithfulness_min
    tox_ok = results["toxicity"] <= settings.toxicity_max
    passed = faith_ok and tox_ok

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "eval_summary.json").write_text(
        json.dumps(
            {
                "n": len(cases),
                "judge": settings.judge_model,
                "scores": results,
                "gates": {
                    "faithfulness_min": settings.faithfulness_min,
                    "toxicity_max": settings.toxicity_max,
                    "faithfulness_pass": faith_ok,
                    "toxicity_pass": tox_ok,
                    "passed": passed,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Eval Summary",
        "",
        f"- Triplets evaluated: **{len(cases)}**",
        f"- Judge model: `{settings.judge_model}` ({settings.llm_backend})",
        f"- **Result: {'✅ PASS' if passed else '❌ FAIL'}**",
        "",
        "| Metric | Score | Gate | Pass |",
        "|---|---|---|---|",
        f"| Faithfulness | {results['faithfulness']:.3f} | ≥ {settings.faithfulness_min} | {'✅' if faith_ok else '❌'} |",
        f"| Toxicity | {results['toxicity']:.3f} | ≤ {settings.toxicity_max} | {'✅' if tox_ok else '❌'} |",
        f"| Answer relevancy | {results['answer_relevancy']:.3f} | ≥ 0.80 (soft) | — |",
        f"| Context recall | {results['contextual_recall']:.3f} | ≥ 0.80 (soft) | — |",
        f"| Context precision | {results['contextual_precision']:.3f} | ≥ 0.70 (soft) | — |",
        "",
    ]
    (REPORTS_DIR / "eval_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"Reports written to {REPORTS_DIR}/")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
