from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def golden_cases():
    from harness.golden import build_test_case, load_golden

    triplets = load_golden()
    if not triplets:
        pytest.skip("golden dataset is empty")
    return [build_test_case(t) for t in triplets]
