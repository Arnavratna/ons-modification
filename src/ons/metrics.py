from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def apply_transaction_costs(
    weights: np.ndarray,
    gross_returns: np.ndarray,
    tc_bps: float = config.TC_BPS,
) -> np.ndarray:
    """
    Deduct one-way transaction costs on turnover at each period.
    Net portfolio return = w·r - tc * ‖Δw‖₁
    """
    T = len(weights)
    net = np.zeros(T)
    for t in range(T):
        port_ret = np.dot(weights[t], gross_returns[t]) - 1
        if t > 0:
            turnover = np.sum(np.abs(weights[t] - weights[t - 1]))
        else:
            turnover = np.sum(np.abs(weights[t] - 1 / weights.shape[1]))
        cost = tc_bps / 10_000 * turnover
        net[t] = port_ret - cost
    return net


def compute_metrics(
    net_rets: np.ndarray,
    gross_returns: np.ndarray,
    weights: np.ndarray,
    dates: pd.DatetimeIndex,
    name: str,
) -> dict:
    """Compute all evaluation metrics for one strategy."""
    del gross_returns

    wealth = np.cumprod(1 + net_rets)
    ann_ret = wealth[-1] ** (252 / len(net_rets)) - 1
    ann_vol = net_rets.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.04) / (ann_vol + 1e-8)
    roll_max = np.maximum.accumulate(wealth)
    dd = (wealth - roll_max) / roll_max
    max_dd = dd.min()
    calmar = ann_ret / (abs(max_dd) + 1e-8)
    turnovers = np.array([
        np.sum(np.abs(weights[t] - weights[t - 1])) if t > 0
        else np.sum(np.abs(weights[t] - 1 / weights.shape[1]))
        for t in range(len(weights))
    ])
    avg_turnover = turnovers.mean() * 252

    return {
        "Name": name,
        "Ann. Return": ann_ret,
        "Ann. Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Calmar Ratio": calmar,
        "Ann. Turnover": avg_turnover,
        "Final Wealth ($)": config.INITIAL_CAPITAL * wealth[-1],
        "_wealth": wealth,
        "_drawdown": dd,
        "_dates": dates,
        "_weights": weights,
        "_net_rets": net_rets,
    }


def compute_regret_vs_bcrp(
    net_rets_strategy: np.ndarray,
    net_rets_bcrp: np.ndarray,
) -> np.ndarray:
    """Cumulative regret = log-wealth(BCRP) - log-wealth(strategy)."""
    log_wealth_s = np.cumsum(np.log1p(net_rets_strategy))
    log_wealth_bcrp = np.cumsum(np.log1p(net_rets_bcrp))
    return log_wealth_bcrp - log_wealth_s
