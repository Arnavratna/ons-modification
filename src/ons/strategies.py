"""Portfolio strategy implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import config
from .utils import project_simplex


def run_vanilla_ons(returns: np.ndarray) -> np.ndarray:
    """
    Vanilla ONS (Hazan, Agarwal, Kale 2007).
    Returns weight matrix (T × N).
    """
    T, N = returns.shape
    delta = config.ONS_DELTA
    beta = config.ONS_BETA

    w = np.ones(N) / N
    A = beta * np.eye(N)
    weights = np.zeros((T, N))

    for t in range(T):
        weights[t] = w
        r = returns[t]
        x = np.clip(np.dot(w, r), 1e-8, None)
        grad = -r / x
        A += np.outer(grad, grad)
        A_inv = np.linalg.inv(A)
        eta = max(1 / (delta * (t + 1)), config.ONS_ETA_FLOOR) if config.ONS_ETA == 0 else config.ONS_ETA
        w_new = w - eta * A_inv @ grad
        w = project_simplex(w_new)

    return weights


def run_ma_ons(returns: np.ndarray, lam: float = config.MA_ONS_LAMBDA) -> np.ndarray:
    """
    Market-Aware ONS: adds L1 turnover penalty λ‖w - w_prev‖₁ to the loss.
    Uses proximal gradient on the Newton direction.
    """
    T, N = returns.shape
    delta = config.ONS_DELTA
    beta = config.ONS_BETA

    w = np.ones(N) / N
    A = beta * np.eye(N)
    weights = np.zeros((T, N))

    for t in range(T):
        weights[t] = w
        r = returns[t]
        x = np.clip(np.dot(w, r), 1e-8, None)
        grad = -r / x
        A += np.outer(grad, grad)
        A_inv = np.linalg.inv(A)
        eta = max(1 / (delta * (t + 1)), config.ONS_ETA_FLOOR) if config.ONS_ETA == 0 else config.ONS_ETA

        w_newton = w - eta * A_inv @ grad

        step_size = np.linalg.norm(w_newton - w) + 1e-10
        threshold = lam * step_size
        diff = w_newton - w
        diff_shrunk = np.sign(diff) * np.maximum(np.abs(diff) - threshold, 0)
        w_prox = w + diff_shrunk

        w = project_simplex(w_prox)

    return weights


def run_eg(returns: np.ndarray) -> np.ndarray:
    """Exponential Gradient portfolio (Kivinen & Warmuth 1997)."""
    T, N = returns.shape
    w = np.ones(N) / N
    weights = np.zeros((T, N))
    for t in range(T):
        weights[t] = w
        r = returns[t]
        x = np.clip(np.dot(w, r), 1e-8, None)
        grad = -r / x
        w_new = w * np.exp(-config.EG_ETA * grad)
        w = w_new / w_new.sum()
    return weights


def run_equal_weight(returns: np.ndarray) -> np.ndarray:
    T, N = returns.shape
    return np.tile(np.ones(N) / N, (T, 1))


def run_bcrp(returns: np.ndarray, tickers: list[str] | None = None) -> np.ndarray:
    """
    Best Constant-Rebalanced Portfolio — hindsight optimal.
    Solved via convex optimisation.
    """
    T, N = returns.shape
    tickers = tickers or config.TICKERS
    w0 = np.ones(N) / N

    def neg_log_wealth(w):
        port_ret = returns @ w
        port_ret = np.clip(port_ret, 1e-8, None)
        return -np.sum(np.log(port_ret))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * N
    res = minimize(
        neg_log_wealth,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    w_star = res.x if res.success else w0
    w_star = np.maximum(w_star, 0)
    w_star /= w_star.sum()
    print(f"  BCRP weights: {dict(zip(tickers[:N], np.round(w_star, 3)))}")
    return np.tile(w_star, (T, 1))


def run_markowitz(returns: np.ndarray, prices: pd.DataFrame | None = None) -> np.ndarray:
    """
    Rolling-window max-Sharpe Markowitz.
    Uses MKZ_WINDOW days of history; holds constant until next rebalance.
    Rebalances monthly (≈ every 21 trading days).
    """
    del prices  # kept for API compatibility with original script

    T, N = returns.shape
    weights = np.zeros((T, N))
    w_current = np.ones(N) / N
    rebal_freq = 21

    for t in range(T):
        if t % rebal_freq == 0 and t >= config.MKZ_WINDOW:
            hist = returns[t - config.MKZ_WINDOW : t]
            mu_hat = hist.mean(axis=0) * 252
            cov_hat = np.cov(hist.T) * 252 + 1e-6 * np.eye(N)
            rf = 0.04

            def neg_sharpe(w):
                ret = w @ mu_hat
                vol = np.sqrt(w @ cov_hat @ w)
                return -(ret - rf) / (vol + 1e-8)

            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
            bounds = [(0, 1)] * N
            res = minimize(
                neg_sharpe,
                w_current,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
            if res.success:
                w_new = np.maximum(res.x, 0)
                w_current = w_new / w_new.sum()

        weights[t] = w_current

    return weights
