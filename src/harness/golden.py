from __future__ import annotations

import json
from pathlib import Path

from harness.config import GOLDEN_DIR

GOLDEN_PATH = GOLDEN_DIR / "golden_dataset.jsonl"


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m harness.generate_golden` first."
        )
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_test_case(triplet: dict):
    from deepeval.test_case import LLMTestCase

    from harness.rag import answer

    result = answer(triplet["question"])
    return LLMTestCase(
        input=triplet["question"],
        actual_output=result.answer,
        expected_output=triplet["ground_truth"],
        retrieval_context=result.contexts,
        context=triplet.get("contexts"),
    )
