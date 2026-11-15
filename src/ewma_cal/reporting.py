"""Cash-separated reference analysis and aligned S&P 500 total-return comparison."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import forecast_metrics, summarize
from .strategy import Strategy

PERIODS = {
    "Full": ("2016-01-04", "2026-11-10"),
    "Development": ("2018-01-01", "2021-12-31"),
    "OOS": ("2022-01-01", "2026-11-10"),
}


def read_benchmark(path, dates):
    raw = pd.read_csv(path, index_col=0, parse_dates=True)
    if "AdjustedClose" not in raw:
        raise ValueError("Benchmark CSV requires Date and AdjustedClose")
    if raw.index.has_duplicates:
        raise ValueError("Duplicate benchmark dates")
    prices = raw.AdjustedClose.reindex(pd.DatetimeIndex(dates))
    if prices.isna().any() or (prices <= 0).any():
        raise ValueError("Missing or nonpositive benchmark values; no forward filling is permitted")
    return prices


def build_report(bundle, config, benchmark_path, output, sizing=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    strategy = Strategy(bundle, config)
    benchmark = read_benchmark(benchmark_path, bundle.dates)
    equities = []
    summaries = []
    forecasts = []
    trade_rows = []
    for hedge in ["strictly_lagged", "post_quote"]:
        for spread in [0.0, 0.25, 0.5, 1.0]:
            if sizing is None:
                result = strategy.simulate(hedge=hedge, spread=spread)
            else:
                from .sizing import replay_sized

                result = replay_sized(strategy, sizing, hedge=hedge, spread=spread)
            for period, (start, end) in PERIODS.items():
                ix, ledger, rf = strategy.period_data(result, start, end)
                dates = bundle.dates[ix]
                spy = config.capital * benchmark.iloc[ix].to_numpy() / benchmark.iloc[ix[0]]
                treasury = config.capital * np.cumprod(1 + rf)
                primary = hedge == config.hedge_timing and spread == 0.0
                if primary:
                    equities.append(
                        pd.DataFrame(
                            dict(
                                Date=dates.strftime("%Y-%m-%d"),
                                Period=period,
                                TradingEquity=ledger.EquityWithoutInterest.to_numpy(),
                                CashEquity=ledger.EquityWithInterest.to_numpy(),
                                CashInterest=ledger.CumulativeCashInterest.to_numpy(),
                                TreasuryEquity=treasury,
                                SP500Equity=spy,
                            )
                        )
                    )
                entered = [
                    r
                    for r in result["Trades"]
                    if bundle.days[ix[0]] <= r["EntryDate"] <= bundle.days[ix[-1]]
                ]
                for name, values in [
                    ("Trading only", ledger.EquityWithoutInterest),
                    ("With cash interest", ledger.EquityWithInterest),
                ]:
                    row = summarize(values, dates, rf, spy)
                    row.update(
                        Period=period,
                        Portfolio=name,
                        Hedge=hedge,
                        SpreadFraction=spread,
                        Trades=len(entered),
                        CashInterest=float(ledger.CumulativeCashInterest.iloc[-1])
                        if name == "With cash interest"
                        else 0.0,
                        Primary=primary,
                        StockHedgeTurnover=sum(r["DollarHedgeTurnover"] for r in entered),
                        MaxMarginProxy=float(result["Margin"][ix].max()),
                        MaxAbsoluteNetVega=float(abs(result["NetVega"][ix]).max()),
                        MaxAbsoluteNetGamma=float(abs(result["NetGamma"][ix]).max()),
                        ExtraCollateralFundingUSD=sum(
                            r.get("ExtraCollateralFundingUSD", 0) for r in entered
                        ),
                        WithinDisplayedDepthFraction=np.mean(
                            [r.get("AllLegsWithinDepth", r["FullDepth"]) for r in entered]
                        ),
                        MeanPositionScale=np.mean([r.get("PositionScale", 1) for r in entered]),
                    )
                    summaries.append(row)
                if primary:
                    for name, values in [
                        ("S&P 500 (SPY total return)", spy),
                        ("Treasury collateral", treasury),
                    ]:
                        row = summarize(values, dates, rf, spy)
                        row.update(
                            Period=period,
                            Portfolio=name,
                            Hedge=hedge,
                            SpreadFraction=spread,
                            Trades=np.nan,
                            CashInterest=np.nan,
                            Primary=True,
                        )
                        summaries.append(row)
                    for book in ["vega", "variance"]:
                        tt = [r for r in entered if r["Book"] == book]
                        forecasts.append(
                            dict(
                                Period=period,
                                Book=book,
                                **forecast_metrics(
                                    [r["NetUSD"] for r in tt],
                                    [r["ForecastMidUSD"] for r in tt],
                                    [r["ForecastRisk"] for r in tt],
                                ),
                            )
                        )
            if hedge == config.hedge_timing and spread == 0.0:
                trade_rows = [
                    {
                        k: r.get(k, np.nan)
                        for k in [
                            "SignalDate",
                            "EntryDate",
                            "ExitDate",
                            "Book",
                            "NetUSD",
                            "ForecastMidUSD",
                            "ForecastRisk",
                            "LastTrainingExit",
                            "MissingDaily",
                            "PositionScale",
                            "ExtraCollateralFundingUSD",
                            "AllLegsWithinDepth",
                        ]
                    }
                    for r in result["Trades"]
                ]
    pd.concat(equities, ignore_index=True).to_csv(output / "equity.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(forecasts).to_csv(output / "forecast_metrics.csv", index=False)
    pd.DataFrame(trade_rows).to_csv(output / "trades.csv", index=False)
    full = pd.concat(equities).query("Period=='Full'").copy()
    full["Year"] = pd.to_datetime(full.Date).dt.year
    annual = full.groupby("Year")[["TradingEquity", "CashEquity", "SP500Equity"]].last()
    previous = annual.shift()
    previous.iloc[0] = config.capital
    (annual / previous - 1).to_csv(output / "annual_returns.csv")
    manifest = dict(
        Model=config.name,
        Version=config.version,
        Hedge=config.hedge_timing,
        Gate=config.vega_gate,
        Capital=config.capital,
        Start=str(bundle.dates[0].date()),
        End=str(bundle.dates[-1].date()),
        Benchmark="SPY adjusted close, total-return proxy for the S&P 500",
        Qualifications=[
            "Reference fills are hypothetical midpoint fills.",
            "Variance sleeve includes fractional option equivalents.",
            "Cash income is collateral interest and not dispersion alpha.",
        ],
        BenchmarkInputSHA256=hashlib.sha256(Path(benchmark_path).read_bytes()).hexdigest(),
    )
    if sizing is not None:
        manifest.update(Model=sizing.name, Version=sizing.version, Sizing=sizing.as_dict())
        manifest["Qualifications"] += [
            "Sizing is clipped to an aggregate margin proxy; this is not a broker margin guarantee.",
            "Extra collateral above initial capital is charged the prior Treasury rate plus the configured funding spread.",
            "Larger midpoint fills and fractional scaled option equivalents are hypothetical.",
            "Payoff predictions precede the additional collateral-funding stress.",
        ]
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return pd.DataFrame(summaries)


def load_reference(root):
    directory = Path(root) / "reports/reference"
    return dict(
        equity=pd.read_csv(directory / "equity.csv", parse_dates=["Date"]),
        metrics=pd.read_csv(directory / "metrics.csv"),
        forecasts=pd.read_csv(directory / "forecast_metrics.csv"),
        annual=pd.read_csv(directory / "annual_returns.csv", index_col=0),
        trades=pd.read_csv(directory / "trades.csv"),
        manifest=json.loads((directory / "manifest.json").read_text()),
    )


def comparison_table(reference, period="Full"):
    f = reference["metrics"]
    return f[(f.Period == period) & f.Primary].set_index("Portfolio")[
        [
            "TotalReturn",
            "CAGR",
            "Sharpe",
            "TreasuryExcessSharpe",
            "MaxDrawdown",
            "Calmar",
            "InformationRatioSP500",
        ]
    ]


def plot_comparison(reference, period="Full"):
    import matplotlib.pyplot as plt

    f = reference["equity"]
    f = f[f.Period == period]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
    for col, label in [
        ("TradingEquity", "Trading only"),
        ("CashEquity", "With cash interest"),
        ("SP500Equity", "S&P 500 (SPY total return)"),
    ]:
        axes[0].plot(f.Date, 100 * (f[col] / f[col].iloc[0] - 1), label=label, lw=1.6)
    axes[0].set(title=f"{period}: cumulative return", ylabel="Return (%)")
    axes[0].legend(fontsize=8)
    axes[1].plot(f.Date, (f.TradingEquity - f.TradingEquity.iloc[0]) / 1000, color="#236192")
    axes[1].set(title="Dispersion trading P&L", ylabel="$ thousands")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.tick_params(axis="x", rotation=20)
    return fig


def plot_drawdowns(reference, period="Full"):
    import matplotlib.pyplot as plt

    f = reference["equity"]
    f = f[f.Period == period]
    fig, ax = plt.subplots(figsize=(11, 3.5), layout="constrained")
    for col, label in [
        ("TradingEquity", "Trading only"),
        ("CashEquity", "With cash interest"),
        ("SP500Equity", "S&P 500 (SPY total return)"),
    ]:
        ax.plot(f.Date, 100 * (f[col] / f[col].cummax() - 1), label=label)
    ax.set(title=f"{period}: drawdown", ylabel="Drawdown (%)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    return fig
