"""Optional public benchmark acquisition; never called during strategy replay."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def download(output, start="2015-12-31", end="2026-11-11"):
    import yfinance as yf

    prices = yf.download(
        "SPY", start=start, end=end, auto_adjust=False, progress=False, threads=False
    )
    if prices.empty:
        raise RuntimeError("SPY provider returned no observations")
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    adjusted = prices[["Adj Close"]].rename(columns={"Adj Close": "AdjustedClose"})
    adjusted.index.name = "Date"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    adjusted.to_csv(output)
    source = dict(
        Symbol="SPY",
        Description="S&P 500 total-return proxy; ETF adjusted close includes distributions and fund expenses",
        Provider="Yahoo Finance via yfinance",
        RetrievedUTC=datetime.now(timezone.utc).isoformat(),
        Start=str(adjusted.index.min().date()),
        End=str(adjusted.index.max().date()),
        Rows=len(adjusted),
    )
    output.with_suffix(".source.json").write_text(json.dumps(source, indent=2) + "\n")
    return adjusted
