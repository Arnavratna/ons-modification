import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ons.strategies import run_equal_weight, run_vanilla_ons  # noqa: E402
from ons.utils import project_simplex  # noqa: E402


def test_project_simplex_sums_to_one() -> None:
    v = np.array([0.5, -0.2, 0.9, 0.1])
    p = project_simplex(v)
    assert np.isclose(p.sum(), 1.0)
    assert (p >= 0).all()


def test_equal_weight_is_constant() -> None:
    returns = np.random.uniform(0.95, 1.05, size=(50, 4))
    weights = run_equal_weight(returns)
    np.testing.assert_allclose(weights, 0.25)


def test_vanilla_ons_stays_on_simplex() -> None:
    returns = np.random.uniform(0.95, 1.05, size=(30, 5))
    weights = run_vanilla_ons(returns)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert (weights >= -1e-10).all()
