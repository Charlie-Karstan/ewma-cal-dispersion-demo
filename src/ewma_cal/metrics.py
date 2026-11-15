"""Portfolio and forecast metrics with explicit benchmarks and missing-mark handling."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def ratio(numerator, denominator):
    return (
        float(numerator / denominator)
        if np.isfinite(denominator) and abs(denominator) > 1e-15
        else np.nan
    )


def summarize(equity, dates, risk_free_returns, benchmark_equity=None):
    values = np.asarray(equity, float)
    dates = pd.DatetimeIndex(dates)
    rf = np.asarray(risk_free_returns, float)
    if len(values) != len(dates) or len(values) != len(rf):
        raise ValueError("Misaligned metric inputs")
    if len(values) < 2 or not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("Need unique chronological observations")
    years = (dates[-1] - dates[0]).days / 365.25
    complete = np.isfinite(values).all()
    daily = values[1:] / values[:-1] - 1
    row = dict(
        Start=str(dates[0].date()),
        End=str(dates[-1].date()),
        Observations=len(values),
        MissingMarks=int((~np.isfinite(values)).sum()),
        StartingEquity=float(values[0]),
        EndingEquity=float(values[-1]),
        NetPnL=float(values[-1] - values[0]),
        TotalReturn=ratio(values[-1], values[0]) - 1,
        CAGR=(values[-1] / values[0]) ** (1 / years) - 1
        if np.isfinite(values[[0, -1]]).all() and min(values[[0, -1]]) > 0
        else np.nan,
    )
    names = [
        "AnnualizedVolatility",
        "Sharpe",
        "TreasuryExcessSharpe",
        "Sortino",
        "MaxDrawdown",
        "Calmar",
        "InformationRatioSP500",
        "BetaSP500",
        "CorrelationSP500",
        "VaR95",
        "ExpectedShortfall95",
        "BestDay",
        "WorstDay",
        "AnnualizedMeanReturn",
    ]
    row.update({k: np.nan for k in names})
    if not complete or not np.isfinite(daily).all():
        return row
    excess = daily - rf[1:]
    std = daily.std(ddof=1)
    dd = values / np.maximum.accumulate(values) - 1
    row.update(
        AnnualizedMeanReturn=float(daily.mean() * 252),
        AnnualizedVolatility=float(std * np.sqrt(252)),
        Sharpe=ratio(np.sqrt(252) * daily.mean(), std),
        TreasuryExcessSharpe=ratio(np.sqrt(252) * excess.mean(), excess.std(ddof=1)),
        Sortino=ratio(np.sqrt(252) * daily.mean(), np.sqrt(np.mean(np.minimum(daily, 0) ** 2))),
        MaxDrawdown=float(dd.min()),
        Calmar=ratio(row["CAGR"], -dd.min()),
        VaR95=float(np.quantile(daily, 0.05)),
        ExpectedShortfall95=float(daily[daily <= np.quantile(daily, 0.05)].mean()),
        BestDay=float(daily.max()),
        WorstDay=float(daily.min()),
    )
    if benchmark_equity is not None:
        benchmark = np.asarray(benchmark_equity, float)
        if len(benchmark) != len(values) or not np.isfinite(benchmark).all():
            raise ValueError("Benchmark must cover every portfolio date")
        b = benchmark[1:] / benchmark[:-1] - 1
        active = daily - b
        row["InformationRatioSP500"] = ratio(np.sqrt(252) * active.mean(), active.std(ddof=1))
        row["BetaSP500"] = ratio(np.cov(daily, b, ddof=1)[0, 1], np.var(b, ddof=1))
        row["CorrelationSP500"] = float(np.corrcoef(daily, b)[0, 1]) if std > 0 else np.nan
    return row


def forecast_metrics(actual, predicted, risk):
    y = np.asarray(actual, float)
    p = np.asarray(predicted, float)
    r = np.asarray(risk, float)
    valid = np.isfinite(y) & np.isfinite(p) & np.isfinite(r) & (r > 0)
    y, p, r = y[valid], p[valid], r[valid]
    if not len(y):
        return dict(
            N=0,
            MSE=np.nan,
            RMSE=np.nan,
            MAE=np.nan,
            ASE=np.nan,
            PearsonIC=np.nan,
            SpearmanIC=np.nan,
            Bias=np.nan,
        )
    error = p - y
    return dict(
        N=len(y),
        MSE=float(np.mean(error**2)),
        RMSE=float(np.sqrt(np.mean(error**2))),
        MAE=float(np.mean(abs(error))),
        ASE=float(np.mean((error / r) ** 2)),
        PearsonIC=float(np.corrcoef(p, y)[0, 1]) if np.std(y) > 0 and np.std(p) > 0 else np.nan,
        SpearmanIC=float(spearmanr(y, p).statistic) if np.std(y) > 0 and np.std(p) > 0 else np.nan,
        Bias=float(error.mean()),
    )
