"""Publication figures from the frozen put model and aggregated diagnostics."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

COLORS = ["#245b83", "#d18428", "#698250"]


def set_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "stix",
            "axes.prop_cycle": plt.cycler(color=COLORS),
            "savefig.dpi": 300,
        }
    )


def correlation_figure(root):
    path = Path(root) / "reports/figures"
    frame = pd.read_csv(path / "correlation_monthly.csv")
    definition = json.loads((path / "correlation_definition.json").read_text())
    dates = pd.to_datetime(frame.Month)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.2),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [1.7, 1]},
    )
    axes[0].plot(dates, 100 * frame.Implied, label="Implied correlation proxy", lw=1.4)
    axes[0].plot(dates, 100 * frame.Realized, label="Realized correlation", lw=1.4)
    axes[0].set(
        title="Implied proxy versus five-session realized correlation", ylabel="Correlation (%)"
    )
    axes[0].legend(frameon=False, fontsize=9)
    mean = 100 * definition["MeanGap"]
    axes[1].plot(dates, 100 * frame.Gap, color=COLORS[0], lw=1.2, label="Implied minus realized")
    axes[1].axhline(
        mean, color=COLORS[1], ls="--", lw=1.4, label=f"Average daily gap: {mean:.2f} pp"
    )
    axes[1].axhline(0, color="#777777", lw=0.7)
    axes[1].set(ylabel="Gap (percentage points)", xlabel="Realization / exit month")
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.18)
    return fig


def payoff_figure(root):
    trades = pd.read_csv(Path(root) / "reports/reference/trades.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2), layout="constrained")
    values = trades[["ForecastMidUSD", "NetUSD"]].to_numpy() / 1000
    lo, hi = np.nanmin(values), np.nanmax(values)
    pad = 0.06 * (hi - lo)
    limits = (lo - pad, hi + pad)
    predicted = trades.ForecastMidUSD.to_numpy() / 1000
    xpad = max(0.06 * np.ptp(predicted), 0.5)
    xlimits = (min(0, predicted.min() - xpad), predicted.max() + xpad)
    for ax, book in zip(axes, ["vega", "variance"]):
        f = trades[trades.Book == book]
        for later, label, color in [
            (False, "Development", COLORS[0]),
            (True, "OOS", COLORS[1]),
        ]:
            selected = f[(f.EntryDate >= "2022-01-01") == later]
            ax.scatter(
                selected.ForecastMidUSD / 1000,
                selected.NetUSD / 1000,
                s=23,
                alpha=0.8,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                label=label,
            )
        ax.plot(limits, limits, color="#666666", ls="--", lw=1, label="Perfect prediction")
        ax.axhline(0, color="#aaaaaa", lw=0.6)
        ax.axvline(0, color="#aaaaaa", lw=0.6)
        ax.set(
            xlim=xlimits,
            ylim=limits,
            aspect="auto",
            xlabel="Predicted payoff ($ thousands)",
            ylabel="Realized net payoff ($ thousands)",
            title=f"{book.title()} sleeve (n = {len(f)})",
        )
        ax.grid(alpha=0.18)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("Predicted versus realized trade payoff — frozen funded put model", fontsize=12)
    return fig


def equity_drawdown_figure(root):
    f = pd.read_csv(Path(root) / "reports/reference/equity.csv", parse_dates=["Date"])
    f = f[f.Period == "Full"]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.4),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [1.4, 1]},
    )
    for col, label in [
        ("TradingEquity", "Dispersion only"),
        ("CashEquity", "Dispersion + cash interest"),
        ("SP500Equity", "S&P 500 (SPY total-return proxy)"),
    ]:
        value = f[col].to_numpy()
        axes[0].plot(f.Date, 100 * value / value[0], label=label, lw=1.4)
        axes[1].plot(f.Date, 100 * (value / np.maximum.accumulate(value) - 1), lw=1.2)
    axes[0].set(
        yscale="log",
        ylabel="Portfolio value (initial = 100; log scale)",
        title="Equity curve and drawdown — frozen funded put model",
    )
    axes[0].set_yticks([100, 150, 200, 300, 400])
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].set(ylabel="Drawdown (%)", xlabel="Date")
    for ax in axes:
        ax.grid(alpha=0.18, which="major")
    return fig


def export_figures(root):
    root = Path(root)
    output = root / "reports/figures"
    output.mkdir(parents=True, exist_ok=True)
    set_style()
    figures = {
        "implied_realized_correlation": correlation_figure(root),
        "predicted_realized_payoff": payoff_figure(root),
        "equity_drawdown": equity_drawdown_figure(root),
    }
    for name, fig in figures.items():
        fig.savefig(
            output / (name + ".pdf"),
            bbox_inches="tight",
            metadata={"CreationDate": None, "ModDate": None},
        )
        fig.savefig(output / (name + ".png"), dpi=300, bbox_inches="tight")
    return figures
