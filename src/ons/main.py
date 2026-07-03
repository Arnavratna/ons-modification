"""Backtest orchestration."""

from __future__ import annotations

import warnings

import pandas as pd

from . import config
from .data import get_prices
from .metrics import apply_transaction_costs, compute_metrics
from .plotting import make_dashboard, make_weights_figure
from .strategies import (
    run_bcrp,
    run_eg,
    run_equal_weight,
    run_ma_ons,
    run_markowitz,
    run_vanilla_ons,
)

warnings.filterwarnings("ignore")


def main() -> None:
    print("=" * 60)
    print("  ONS Portfolio Backtesting Simulation")
    print("=" * 60)

    prices = get_prices()
    N = prices.shape[1]

    ret_relatives = (prices / prices.shift(1)).dropna().values
    dates_used = prices.index[1:]

    T = len(ret_relatives)
    print(
        f"\nBacktest: {dates_used[0].date()} → {dates_used[-1].date()}"
        f"  ({T} days, {N} assets)\n"
    )

    print("Running Vanilla ONS …")
    w_ons = run_vanilla_ons(ret_relatives)

    print("Running Market-Aware ONS …")
    w_ma_ons = run_ma_ons(ret_relatives, lam=config.MA_ONS_LAMBDA)

    print("Running BCRP (convex optimisation) …")
    w_bcrp = run_bcrp(ret_relatives, tickers=list(prices.columns))

    print("Running Exponential Gradient …")
    w_eg = run_eg(ret_relatives)

    print("Running Equal Weight …")
    w_eq = run_equal_weight(ret_relatives)

    print("Running Markowitz …")
    w_mkz = run_markowitz(ret_relatives, prices)

    strategies = [
        ("Vanilla ONS", w_ons),
        ("MA-ONS", w_ma_ons),
        ("BCRP", w_bcrp),
        ("Exp. Gradient", w_eg),
        ("Equal Weight", w_eq),
        ("Markowitz", w_mkz),
    ]

    results = []
    print(f"\nComputing net returns with {config.TC_BPS:.0f} bps transaction costs …")
    for name, weights in strategies:
        net_rets = apply_transaction_costs(weights, ret_relatives, config.TC_BPS)
        metrics = compute_metrics(net_rets, ret_relatives, weights, dates_used, name)
        results.append(metrics)

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    header = f"{'Strategy':<18}{'Ann.Ret':>9}{'Vol':>9}{'Sharpe':>9}"
    header += f"{'MaxDD':>10}{'Calmar':>9}{'Turnover':>11}{'FinalVal':>14}"
    print(header)
    print("-" * 90)
    for r in results:
        row = (
            f"{r['Name']:<18}"
            f"{r['Ann. Return'] * 100:>8.2f}%"
            f"{r['Ann. Volatility'] * 100:>8.2f}%"
            f"{r['Sharpe Ratio']:>9.2f}"
            f"{r['Max Drawdown'] * 100:>9.2f}%"
            f"{r['Calmar Ratio']:>9.2f}"
            f"{r['Ann. Turnover']:>10.1f}x"
            f"  ${r['Final Wealth ($)']:>12,.0f}"
        )
        print(row)

    summary_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    pd.DataFrame(summary_rows).to_csv("ons_metrics.csv", index=False)
    print("\nMetrics saved → ons_metrics.csv")

    print("Building dashboard …")
    make_dashboard(results)

    print("Building weights figure …")
    make_weights_figure(results, list(prices.columns))

    print("\nDone.")
