"""Shared numerical utilities."""

import numpy as np


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Project onto the probability simplex Δ^n."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u > (cssv - 1) / np.arange(1, n + 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)
