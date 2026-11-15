"""Build private replay inputs from an existing licensed local archive."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import market
from .actions import announced_actions
from .covariance import estimate_covariances
from .data import MarketBundle, read_dividend_announcements
from .greeks import straddle
from .hedging import build_units
from .opportunities import generate_variance_opportunities, generate_vega_opportunities


def prepare(data_root, output, end="2026-11-10"):
    data_root = Path(data_root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    members = market.load_members(data_root)
    days = sorted(d for d in members if "2016-01-01" <= d <= end)
    source = (
        pd.read_csv(data_root / "raw/dow/rates/dgs1mo.csv", dtype={"Date": str})
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
    )
    rates = (
        source.set_index(pd.to_datetime(source.Date))
        .RateDecimal.astype(float)
        .reindex(pd.to_datetime(days), method="ffill")
    )
    if rates.isna().any():
        raise ValueError("Missing initial risk-free rate")
    calendar = dict(Members=members, Days=days, Rates=dict(zip(days, rates)))
    returns = pd.read_csv(
        data_root / "processed/dow/dow_total_log_returns.csv", index_col=0, parse_dates=True
    )
    archive = market.Archive(data_root)
    selection = SimpleNamespace(min_dte=14, max_dte=70, target_dte=30)
    snapshots = {}
    for n, day in enumerate(days):
        if day < "2017-01-01":
            continue
        rows = []
        expiry = None
        for ric in ["DIA.P", *members[day].RIC.tolist()]:
            symbol = market.trading_symbol(ric, day)
            pairs = market.choose_pair(
                archive.chain(day, symbol), day, selection, expiry, symbol, ranked=True
            )
            if not pairs:
                break
            near = [p for p in pairs if p[2] == pairs[0][2]][:2]
            pair = min(near, key=lambda p: sum(float(q["ask"]) - float(q["bid"]) for q in p[4:6]))
            expiry = pair[2]
            try:
                m = straddle(pair, day, calendar["Rates"][day])
            except ValueError:
                break
            tau = (pd.Timestamp(expiry) - pd.Timestamp(day)).days / 365.25
            forward = pair[3] + (market.midpoint(pair[4]) - market.midpoint(pair[5])) / np.exp(
                -calendar["Rates"][day] * tau
            )
            rows.append(
                dict(
                    RIC=ric,
                    Spot=pair[6],
                    IV=m["iv"],
                    Vega=m["vega"],
                    Gamma=m["gamma"],
                    Delta=m["delta"],
                    HalfSpread=m["spread"],
                    Premium=100 * sum(market.midpoint(q) for q in pair[4:6]),
                    Strike=pair[3],
                    Call=pair[4]["optionSymbol"],
                    Put=pair[5]["optionSymbol"],
                    Expiry=expiry,
                    Rate=calendar["Rates"][day],
                    QYield=calendar["Rates"][day] - np.log(forward / pair[6]) / tau,
                    Weight=0
                    if ric == "DIA.P"
                    else float(members[day].set_index("RIC").loc[ric, "Weight"]),
                    Volume=min(float(q.get("volume") or 0) for q in pair[4:6]),
                    OI=min(float(q.get("openInterest") or 0) for q in pair[4:6]),
                    Pair=pair,
                )
            )
        if len(rows) == 31:
            snapshots[day] = pd.DataFrame(rows)
        if n % 250 == 0:
            print(f"Snapshots: {n}/{len(days)}", flush=True)
    covariances = estimate_covariances(snapshots, returns)
    print("Building post-quote and lagged hedge paths", flush=True)
    units = {
        "post_quote": build_units(data_root, snapshots, calendar),
        "strictly_lagged": build_units(data_root, snapshots, calendar, lagged=True),
    }
    bundle = MarketBundle(
        calendar,
        snapshots,
        units,
        {},
        covariances,
        returns,
        announced_actions(data_root),
        read_dividend_announcements(data_root),
    )
    for book, fn in [
        ("vega", generate_vega_opportunities),
        ("variance", generate_variance_opportunities),
    ]:
        print(f"Generating {book} shadow history", flush=True)
        h = fn(bundle)
        bundle.histories["baseline_" + book] = h
        h.to_pickle(output / (book + "_shadow_trades.pkl"))
    for name, value in [
        ("calendar", calendar),
        ("snapshots", snapshots),
        ("units", units["post_quote"]),
        ("units_lagged", units["strictly_lagged"]),
        ("covariances", covariances),
    ]:
        pd.to_pickle(value, output / (name + ".pkl"))
    (output / "announced_actions.json").write_text(json.dumps(bundle.actions, indent=2))
    (output / "dividends.json").write_text(json.dumps(bundle.dividends, indent=2))
    return bundle
