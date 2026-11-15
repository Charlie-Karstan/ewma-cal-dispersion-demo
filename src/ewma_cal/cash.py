"""Interest accounting for the separately financed, frozen trading overlay.

Capital collateral earns the prior observed one-month Treasury yield on ACT/365.25.
Existing trading funding charges stay in P&L. Only settled net trade P&L changes
cash; unrealized marks earn no interest. Interest is reinvested and cannot resize
positions or change signals. This is an explicit collateral-remuneration assumption.
"""

import numpy as np
import pandas as pd

from .accounting import monthly_statistics


def cash_interest_overlay(equity_without_interest, dates, annual_rates, realized_pnl, capital):
    dates = pd.DatetimeIndex(dates)
    base = np.asarray(equity_without_interest, dtype=float)
    if len(base) != len(dates) or len(dates) == 0:
        raise ValueError("Equity and dates must have matching nonempty lengths")
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Valuation dates must be unique and increasing")
    rates = pd.Series(annual_rates, dtype=float)
    rates.index = pd.to_datetime(rates.index)
    if rates.index.has_duplicates:
        raise ValueError("Duplicate rate dates")
    aligned = rates.sort_index().reindex(dates, method="ffill").to_numpy()
    if not np.isfinite(aligned).all():
        raise ValueError("Missing as-of Treasury rates")
    flows = pd.Series(realized_pnl, dtype=float)
    flows.index = pd.to_datetime(flows.index)
    flows = flows.groupby(level=0).sum().reindex(dates, fill_value=0).to_numpy()
    credits = np.zeros(len(dates))
    rf_returns = np.zeros(len(dates))
    balances = np.zeros(len(dates))
    balance = float(capital)
    for k in range(len(dates)):
        if k:
            dt = (dates[k] - dates[k - 1]).days / 365.25
            rf_returns[k] = aligned[k - 1] * dt
            credits[k] = balance * rf_returns[k]
        balance += credits[k] + flows[k]
        if balance < 0:
            raise ValueError("Collateral cash depleted; no new borrowing assumption is permitted")
        balances[k] = balance
    cumulative = credits.cumsum()
    return pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "EquityWithoutInterest": base,
            "EquityWithInterest": base + cumulative,
            "SettledCollateralCash": balances,
            "RealizedTradingPnL": flows,
            "CashInterestCredit": credits,
            "CumulativeCashInterest": cumulative,
            "RiskFreeIntervalReturn": rf_returns,
        }
    )


def portfolio_kpis(equity, dates, capital, risk_free_returns):
    values = np.asarray(equity, dtype=float)
    dates = pd.DatetimeIndex(dates)
    daily = values[1:] / values[:-1] - 1
    excess = daily - np.asarray(risk_free_returns, dtype=float)[1:]
    years = (dates[-1] - dates[0]).days / 365.25

    def sharpe(returns):
        sd = np.std(returns, ddof=1)
        return float(np.sqrt(252) * np.mean(returns) / sd) if sd > 0 else float("nan")

    return {
        "Capital": float(capital),
        "NetUSD": float(values[-1] - capital),
        "TotalReturn": float(values[-1] / capital - 1),
        "CAGR": float((values[-1] / capital) ** (1 / years) - 1),
        "DailySharpeZeroBenchmark": sharpe(daily),
        "DailySharpeTreasuryBenchmark": sharpe(excess),
        "MonthlySharpeZeroBenchmark": float(monthly_statistics(values, dates)[0]),
        "AnnualizedVolatility": float(np.std(daily, ddof=1) * np.sqrt(252)),
        "MaxDrawdown": float(
            np.min(values / np.maximum.accumulate(np.r_[capital, values])[1:] - 1)
        ),
    }
