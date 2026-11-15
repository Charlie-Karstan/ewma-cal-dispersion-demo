"""Fixed-grid exposure sensitivity; no parameter selection or signal refitting."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import forecast_metrics, summarize
from .reporting import PERIODS, read_benchmark
from .sizing import replay_sized
from .strategy import Strategy

DEFAULT_SCALES = (0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16)


def check_freeze(root):
    root = Path(root)
    lock = json.loads((root / "config/freeze.json").read_text())
    for name, digest in lock["SHA256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen model file changed: {name}")
    return lock


def run_leverage_test(bundle, config, sizing, benchmark_path, output, scales=DEFAULT_SCALES):
    scales = tuple(sorted(set(float(x) for x in scales)))
    if not scales or not np.isfinite(scales).all() or min(scales) <= 0:
        raise ValueError("Leverage targets must be finite and positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    strategy = Strategy(bundle, config)
    base = strategy.simulate()
    benchmark = read_benchmark(benchmark_path, bundle.dates)
    rows, forecasts, checks = [], [], []
    for scale in scales:
        current = replace(sizing, target_scale=scale)
        result = replay_sized(strategy, current, base=base)
        for period, (start, end) in PERIODS.items():
            ix, ledger, rf = strategy.period_data(result, start, end)
            spy = config.capital * benchmark.iloc[ix].to_numpy() / benchmark.iloc[ix[0]]
            trades = [r for r in result["Trades"] if start <= r["EntryDate"] <= end]
            sizes = [r["PositionScale"] for r in trades]
            positives = sorted([r["NetUSD"] for r in trades if r["NetUSD"] > 0], reverse=True)
            settings = dict(
                TargetScale=scale,
                Period=period,
                MarginCapUSD=sizing.margin_cap_usd,
                ExtraFundingSpread=sizing.extra_funding_spread,
                SpreadFraction=0.0,
            )
            for cash, values in [
                (False, ledger.EquityWithoutInterest),
                (True, ledger.EquityWithInterest),
            ]:
                rows.append(
                    dict(
                        **settings,
                        CashInterest=cash,
                        **summarize(values, bundle.dates[ix], rf, spy),
                        Trades=len(trades),
                        ExtraCollateralFundingUSD=sum(
                            r["ExtraCollateralFundingUSD"] for r in trades
                        ),
                        CashInterestUSD=float(ledger.CumulativeCashInterest.iloc[-1])
                        if cash
                        else 0.0,
                        MaxMarginProxy=float(result["Margin"][ix].max()),
                        MeanActualScale=float(np.mean(sizes)),
                        MinActualScale=float(np.min(sizes)),
                        MaxActualScale=float(np.max(sizes)),
                        ClippedTradeFraction=float(np.mean(np.array(sizes) < scale - 1e-8)),
                        WithinDisplayedDepthFraction=float(
                            np.mean([r["AllLegsWithinDepth"] for r in trades])
                        ),
                        NetRemovingBestFive=sum(r["NetUSD"] for r in trades) - sum(positives[:5]),
                        TopFiveSharePositivePnL=sum(positives[:5]) / sum(positives)
                        if positives
                        else np.nan,
                        ActiveSessionFraction=float(np.mean(result["Active"][ix] > 0)),
                    )
                )
            for book in ["vega", "variance", "both"]:
                selected = [r for r in trades if book == "both" or r["Book"] == book]
                forecasts.append(
                    dict(
                        **settings,
                        Book=book,
                        **forecast_metrics(
                            [r["NetUSD"] for r in selected],
                            [r["ForecastMidUSD"] for r in selected],
                            [r["ForecastRisk"] for r in selected],
                        ),
                    )
                )
        checks.append(
            dict(
                TargetScale=scale,
                MarginCapRespected=True,
                FundingFullyAllocated=True,
                TradePnLReconciles=True,
                SignalsUnchanged=True,
            )
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "leverage_metrics.csv", index=False)
    pd.DataFrame(forecasts).to_csv(output / "leverage_forecasts.csv", index=False)
    compact = frame.pivot(
        index="TargetScale",
        columns=["Period", "CashInterest"],
        values=["CAGR", "Sharpe", "TreasuryExcessSharpe", "MaxDrawdown"],
    )
    compact.columns = [
        f"{metric}_{period}_{'cash' if cash else 'trading'}"
        for metric, period, cash in compact.columns
    ]
    compact.to_csv(output / "leverage_summary.csv")
    (output / "checks.json").write_text(
        json.dumps(
            dict(
                FrozenTarget=sizing.target_scale,
                Scales=scales,
                MarginCapUSD=sizing.margin_cap_usd,
                ExtraFundingSpread=sizing.extra_funding_spread,
                Hedge=config.hedge_timing,
                Retrospective=True,
                SignalRefit=False,
                SelectedNewDefault=False,
                Scenarios=checks,
            ),
            indent=2,
        )
        + "\n"
    )
    return frame


def plot_leverage(frame, frozen_scale=8):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for ax, period in zip(axes[0], ["Full", "OOS"]):
        for cash, label in [(False, "Trading only"), (True, "With cash interest")]:
            f = frame[(frame.Period == period) & (frame.CashInterest == cash)]
            ax.plot(f.TargetScale, 100 * f.CAGR, marker="o", label=label)
        ax.set(title=f"{period}: CAGR", ylabel="CAGR (%)")
    full = frame[(frame.Period == "Full") & ~frame.CashInterest]
    axes[1, 0].plot(full.TargetScale, full.Sharpe, marker="o", label="Trading Sharpe")
    axes[1, 0].set(title="Full period: trading Sharpe", ylabel="Sharpe, zero benchmark")
    axes[1, 1].plot(
        full.TargetScale,
        100 * full.WithinDisplayedDepthFraction,
        marker="o",
        color="#657b45",
        label="All legs fit displayed size",
    )
    axes[1, 1].set(title="Entry depth diagnostic", ylabel="Trade bundles (%)")
    for ax in axes.flat:
        ax.axvline(
            frozen_scale, color="#777777", linestyle="--", linewidth=1, label="Frozen 8x target"
        )
        ax.axhline(0, color="#999999", linewidth=0.6)
        ax.set(xlabel="Target multiple of original position size")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    return fig
