from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .metrics import compute_regret_vs_bcrp


def _save_and_reveal(fig: plt.Figure, filename: str) -> Path:
    path = Path(filename).resolve()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nSaved → {path}")
    if config.AUTO_OPEN_PLOTS and sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    elif config.SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
    return path

COLOURS = {
    "Vanilla ONS": "#FF4500",
    "MA-ONS": "#00FFCC",
    "BCRP": "#FFD700",
    "Exp. Gradient": "#FF69B4",
    "Equal Weight": "#1E90FF",
    "Markowitz": "#ADFF2F",
}


def make_dashboard(results: list[dict]) -> None:
    fig = plt.figure(figsize=(22, 28), facecolor="#0F1117")
    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.45, wspace=0.30)

    ax_wealth = fig.add_subplot(gs[0, :])
    ax_dd = fig.add_subplot(gs[1, :])
    ax_regret = fig.add_subplot(gs[2, 0])
    ax_weights = fig.add_subplot(gs[2, 1])
    ax_metrics = fig.add_subplot(gs[3, :])
    ax_turn = fig.add_subplot(gs[4, 0])
    ax_sharpe = fig.add_subplot(gs[4, 1])

    for ax in [ax_wealth, ax_dd, ax_regret, ax_weights, ax_metrics, ax_turn, ax_sharpe]:
        ax.set_facecolor("#181C25")
        ax.tick_params(colors="#CCCCCC", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333344")

    ax_wealth.set_title(
        f"Portfolio Wealth  (net of {config.TC_BPS} bps transaction costs)",
        color="white",
        fontsize=13,
        pad=10,
    )
    for r in results:
        ax_wealth.plot(
            r["_dates"],
            config.INITIAL_CAPITAL * r["_wealth"],
            label=r["Name"],
            color=COLOURS.get(r["Name"], "grey"),
            linewidth=1.8,
        )
    ax_wealth.axhline(config.INITIAL_CAPITAL, color="#555566", lw=0.7, ls="--")
    ax_wealth.set_ylabel("Portfolio Value ($)", color="#CCCCCC", fontsize=10)
    ax_wealth.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_wealth.legend(loc="upper left", framealpha=0.2, labelcolor="white", fontsize=9)

    ax_dd.set_title("Drawdown from Peak", color="white", fontsize=13, pad=10)
    for r in results:
        ax_dd.fill_between(
            r["_dates"],
            r["_drawdown"] * 100,
            0,
            alpha=0.35,
            color=COLOURS.get(r["Name"], "grey"),
        )
        ax_dd.plot(
            r["_dates"],
            r["_drawdown"] * 100,
            label=r["Name"],
            color=COLOURS.get(r["Name"], "grey"),
            linewidth=1.2,
        )
    ax_dd.set_ylabel("Drawdown (%)", color="#CCCCCC", fontsize=10)
    ax_dd.legend(loc="lower left", framealpha=0.2, labelcolor="white", fontsize=9)

    ax_regret.set_title(
        "Cumulative Regret vs BCRP\n(lower = closer to hindsight optimal)",
        color="white",
        fontsize=11,
        pad=8,
    )
    bcrp_res = next(r for r in results if r["Name"] == "BCRP")
    for r in results:
        if r["Name"] == "BCRP":
            continue
        regret = compute_regret_vs_bcrp(r["_net_rets"], bcrp_res["_net_rets"])
        ax_regret.plot(
            r["_dates"],
            regret,
            label=r["Name"],
            color=COLOURS.get(r["Name"], "grey"),
            linewidth=1.5,
        )
    ax_regret.axhline(0, color="white", lw=0.6, ls="--")
    ax_regret.set_ylabel("Log-Wealth Regret", color="#CCCCCC", fontsize=10)
    ax_regret.legend(framealpha=0.2, labelcolor="white", fontsize=9)

    ax_weights.set_title("Cumulative Net Return by Strategy", color="white", fontsize=11, pad=8)
    for r in results:
        cum = np.cumprod(1 + r["_net_rets"])
        ax_weights.plot(
            r["_dates"],
            (cum - 1) * 100,
            label=r["Name"],
            color=COLOURS.get(r["Name"], "grey"),
            linewidth=1.5,
        )
    ax_weights.axhline(0, color="#555566", lw=0.7, ls="--")
    ax_weights.set_ylabel("Cumulative Return (%)", color="#CCCCCC", fontsize=10)
    ax_weights.legend(framealpha=0.2, labelcolor="white", fontsize=8)

    ax_metrics.axis("off")
    ax_metrics.set_title("Performance Metrics Summary", color="white", fontsize=13, pad=10)
    col_labels = [
        "Strategy", "Ann. Return", "Volatility", "Sharpe",
        "Max DD", "Calmar", "Ann. Turnover", "Final Value",
    ]
    table_data = []
    for r in results:
        table_data.append([
            r["Name"],
            f"{r['Ann. Return'] * 100:.2f}%",
            f"{r['Ann. Volatility'] * 100:.2f}%",
            f"{r['Sharpe Ratio']:.2f}",
            f"{r['Max Drawdown'] * 100:.2f}%",
            f"{r['Calmar Ratio']:.2f}",
            f"{r['Ann. Turnover']:.1f}x",
            f"${r['Final Wealth ($)']:,.0f}",
        ])
    tbl = ax_metrics.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.0)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#444466")
        if row == 0:
            cell.set_facecolor("#2A2D3E")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            strat_name = table_data[row - 1][0]
            cell.set_facecolor("#181C25")
            cell.set_text_props(color=COLOURS.get(strat_name, "#CCCCCC"))

    ax_turn.set_title("Annualised Portfolio Turnover", color="white", fontsize=11, pad=8)
    names = [r["Name"] for r in results]
    turns = [r["Ann. Turnover"] for r in results]
    bars = ax_turn.bar(
        names,
        turns,
        color=[COLOURS.get(n, "grey") for n in names],
        edgecolor="#333344",
        alpha=0.85,
    )
    ax_turn.set_ylabel("Turnover (×/year)", color="#CCCCCC", fontsize=10)
    ax_turn.tick_params(axis="x", rotation=20)
    for bar, val in zip(bars, turns):
        ax_turn.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.1f}×",
            ha="center",
            va="bottom",
            color="white",
            fontsize=8,
        )

    ax_sharpe.set_title("Sharpe Ratio (rf = 4%)", color="white", fontsize=11, pad=8)
    sharpes = [r["Sharpe Ratio"] for r in results]
    bars2 = ax_sharpe.bar(
        names,
        sharpes,
        color=[COLOURS.get(n, "grey") for n in names],
        edgecolor="#333344",
        alpha=0.85,
    )
    ax_sharpe.axhline(0, color="white", lw=0.6, ls="--")
    ax_sharpe.set_ylabel("Sharpe Ratio", color="#CCCCCC", fontsize=10)
    ax_sharpe.tick_params(axis="x", rotation=20)
    for bar, val in zip(bars2, sharpes):
        ypos = bar.get_height() + 0.01 if val >= 0 else bar.get_height() - 0.12
        ax_sharpe.text(
            bar.get_x() + bar.get_width() / 2,
            ypos,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            color="white",
            fontsize=8,
        )

    fig.suptitle(
        "ONS Portfolio Optimisation — Backtesting Dashboard",
        color="white",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    _save_and_reveal(fig, "ons_backtest_dashboard.png")


def make_weights_figure(results: list[dict], tickers: list[str]) -> None:
    n_strats = len(results)
    cols = 2
    rows = (n_strats + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 5), facecolor="#0F1117")
    axes = axes.flatten()
    asset_colors = plt.cm.tab20(np.linspace(0, 1, len(tickers)))

    for i, r in enumerate(results):
        ax = axes[i]
        ax.set_facecolor("#181C25")
        ax.tick_params(colors="#CCCCCC", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333344")

        w = r["_weights"]
        dates = r["_dates"]
        n_assets = w.shape[1]
        ticker_labels = tickers[:n_assets]

        w_df = pd.DataFrame(w, index=dates, columns=ticker_labels)
        w_monthly = w_df.resample("ME").last()

        ax.stackplot(
            w_monthly.index,
            w_monthly.T.values,
            labels=ticker_labels,
            colors=asset_colors[:n_assets],
            alpha=0.88,
        )
        ax.set_title(r["Name"], color="white", fontsize=12, pad=8)
        ax.set_ylabel("Weight", color="#CCCCCC", fontsize=9)
        ax.set_ylim(0, 1)
        ax.legend(
            loc="upper right",
            framealpha=0.15,
            labelcolor="white",
            fontsize=6,
            ncol=3,
            borderpad=0.4,
        )

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Portfolio Weight Evolution — All Strategies",
        color="white",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    _save_and_reveal(fig, "ons_weights_all.png")
