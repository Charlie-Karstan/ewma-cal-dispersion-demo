"""Exact-contract option marks and explicitly timed stock hedges."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import market as r
from .actions import announced_actions
from .greeks import entry_pair, straddle


def build_units(data_root, snaps, calendar, lagged=False):
    data_root = Path(data_root)
    cal = calendar
    days = cal["Days"]
    rates = cal["Rates"]
    loc = {d: i for i, d in enumerate(days)}
    returns = pd.read_csv(
        data_root / "processed/dow/dow_total_log_returns.csv", index_col=0, parse_dates=True
    )
    archive = r.Archive(data_root)
    announced = announced_actions(data_root)
    actions = {ric: sorted(set(d for ann, d in events)) for ric, events in announced.items()}
    divs = {}
    for file in (data_root / "raw/dow/dividends").glob("*.csv"):
        z = pd.read_csv(file)
        if "Dividend Ex Date" not in z:
            continue
        divs[file.stem] = [
            (
                str(x["Dividend Ex Date"])[:10],
                str(x["Dividend Announcement Date"])[:10],
                float(x["Gross Dividend Amount"]),
            )
            for _, x in z.iterrows()
            if pd.notna(x["Dividend Ex Date"]) and pd.notna(x["Gross Dividend Amount"])
        ]
    prices = {}
    for ric in returns:
        stem = ric.replace(".", "_").replace("^", "_")
        file = data_root / "raw/dow/prices" / (stem + ".csv")
        rawfile = file.with_name(stem + "_unadjusted.csv")
        if rawfile.exists():
            file = rawfile
        if file.exists():
            z = pd.read_csv(file)
            column = next(
                (
                    c
                    for c in ["Close Price", "TRDPRC_1", "UnadjustedClose", "Close", "Price"]
                    if c in z
                ),
                None,
            )
            if column is not None:
                prices[ric] = dict(
                    zip(z.Date.astype(str), pd.to_numeric(z[column], errors="coerce"))
                )

    @lru_cache(maxsize=200000)
    def quote(c, p, day):
        cq, pq = archive.history(c).get(day), archive.history(p).get(day)
        return cq, pq

    units = {}
    for n, (signal, snapshot) in enumerate(snaps.items()):
        i = loc[signal]
        if i + 2 >= len(days):
            continue
        entry = days[i + 1]
        for row in snapshot.itertuples():
            types = ["straddle", "put"] if row.RIC == "DIA.P" else ["straddle"]
            cq, pq = quote(row.Call, row.Put, entry)
            if cq is None or pq is None:
                continue
            pair = entry_pair(row.Pair, cq, pq)
            if pair is None:
                continue
            try:
                initial = straddle(pair, entry, rates[entry])
            except ValueError:
                continue
            for instrument in types:
                cq, pq = quote(row.Call, row.Put, entry)
                pair = entry_pair(row.Pair, cq, pq)
                side = "P" if instrument == "put" else "CP"
                fee = 1.3 if side == "P" else 2.6
                selected = [pq] if side == "P" else [cq, pq]
                p0 = 100 * sum(r.midpoint(x) for x in selected)
                s0 = pair[6]
                hs0 = 50 * sum(float(x["ask"]) - float(x["bid"]) for x in selected)

                def delta(metric, pair):
                    if side == "CP":
                        return metric["delta"]
                    tau = (
                        pd.Timestamp(row.Expiry) - pd.Timestamp(pair[4].get("_date", entry))
                    ).days / 365.25
                    forward_disc = (
                        pair[3] * np.exp(-rates[entry] * max(tau, 0))
                        + r.midpoint(pair[4])
                        - r.midpoint(pair[5])
                    )
                    return 0.5 * (metric["delta"] - 100 * forward_disc)

                first_delta = (
                    initial["delta"]
                    if side == "CP"
                    else 0.5
                    * (
                        initial["delta"]
                        - 100
                        * (
                            row.Strike
                            * np.exp(
                                -rates[entry]
                                * (pd.Timestamp(row.Expiry) - pd.Timestamp(entry)).days
                                / 365.25
                            )
                            + r.midpoint(cq)
                            - r.midpoint(pq)
                        )
                    )
                )
                pending_shares = round(-first_delta / s0)
                signal_delta = (
                    row.Delta
                    if side == "CP"
                    else 0.5
                    * (
                        row.Delta
                        - 100
                        * (
                            row.Strike
                            * np.exp(
                                -rates[signal]
                                * (pd.Timestamp(row.Expiry) - pd.Timestamp(signal)).days
                                / 365.25
                            )
                            + r.midpoint(row.Pair[4])
                            - r.midpoint(row.Pair[5])
                        )
                    )
                )
                shares = round(-signal_delta / row.Spot) if lagged else pending_shares
                hg = hc_long = hc_short = 0.0
                long_years = short_years = 0.0
                exec_cost = abs(shares) * (s0 * 0.0001 + 0.005)
                turnover = abs(shares) * s0
                share_turnover = abs(shares)
                prevspot = s0
                hair = 0.0
                hedge_unknown = False
                long_exposure_years = [0.0]
                short_exposure_years = [0.0]
                share_turnovers = [2 * abs(shares)]
                dates = [entry]
                mid = [[-fee - 2 * exec_cost, -fee - 2 * exec_cost]]
                spread = [2 * hs0]
                hedge_turn = [2 * turnover]
                complete = [True]
                divlist = divs.get(row.RIC.replace(".", "_").replace("^", "_"), [])
                for j in range(i + 2, len(days)):
                    day = days[j]
                    if (pd.Timestamp(row.Expiry) - pd.Timestamp(day)).days < 1:
                        break
                    cq, pq = quote(row.Call, row.Put, day)
                    markable = cq is not None and pq is not None
                    pair = entry_pair(row.Pair, cq, pq) if markable else None
                    if pair is None:
                        markable = False
                    actual_event = any(entry < d <= day for d in actions.get(row.RIC, []))
                    logret = returns[row.RIC].get(pd.Timestamp(day), np.nan)
                    grossdiv = sum(amount for ex, ann, amount in divlist if ex == day)
                    spot = pair[6] if markable else prices.get(row.RIC, {}).get(day, np.nan)
                    if not np.isfinite(spot) and np.isfinite(logret):
                        spot = prevspot * np.exp(logret) - grossdiv
                    valid_stock = (
                        np.isfinite(spot)
                        and spot > 0
                        and np.isfinite(logret)
                        and abs(prevspot * np.exp(logret) - spot) < 0.08 * prevspot
                    )
                    dt = (pd.Timestamp(day) - pd.Timestamp(days[j - 1])).days / 365.25
                    if valid_stock:
                        long_years += dt * max(shares, 0) * prevspot
                        short_years += dt * max(-shares, 0) * prevspot
                        hg += shares * prevspot * np.expm1(logret)
                        hc_long += dt * (
                            max(shares, 0) * prevspot * max(rates[days[j - 1]] + 0.03, 0)
                            + max(-shares, 0) * prevspot * 0.01
                        )
                        hc_short += dt * (
                            max(-shares, 0) * prevspot * max(rates[days[j - 1]] + 0.03, 0)
                            + max(shares, 0) * prevspot * 0.01
                        )
                    else:
                        actual_event = True
                        hedge_unknown = True
                    actual_event = actual_event or hedge_unknown
                    liquidation = (
                        exec_cost + abs(shares) * (spot * 0.0001 + 0.005) if valid_stock else np.nan
                    )
                    total_turnover = turnover + abs(shares) * spot if valid_stock else np.nan
                    if side == "CP" and grossdiv and prevspot > row.Strike:
                        hair += 100 * grossdiv
                    if markable and valid_stock and not actual_event:
                        selected = [pq] if side == "P" else [cq, pq]
                        premium = 100 * sum(r.midpoint(x) for x in selected)
                        hs = 50 * sum(float(x["ask"]) - float(x["bid"]) for x in selected)
                        funding = (
                            p0
                            * rates[entry]
                            * (pd.Timestamp(day) - pd.Timestamp(entry)).days
                            / 365.25
                        )
                        mid.append(
                            [
                                premium - p0 + hg - liquidation - hc_long - fee - funding,
                                -(premium - p0) - hg - liquidation - hc_short - fee - hair,
                            ]
                        )
                        spread.append(hs0 + hs)
                        complete.append(True)
                    else:
                        mid.append([np.nan, np.nan])
                        spread.append(np.nan)
                        complete.append(False)
                    dates.append(day)
                    hedge_turn.append(total_turnover)
                    long_exposure_years.append(long_years)
                    short_exposure_years.append(short_years)
                    share_turnovers.append(share_turnover + abs(shares))
                    # Rebalance AFTER this day's realized return and valuation, never before it.
                    if markable and valid_stock and not actual_event:
                        try:
                            metric = straddle(pair, day, rates[day])
                            current_delta = (
                                metric["delta"]
                                if side == "CP"
                                else 0.5
                                * (
                                    metric["delta"]
                                    - 100
                                    * (
                                        row.Strike
                                        * np.exp(
                                            -rates[day]
                                            * (pd.Timestamp(row.Expiry) - pd.Timestamp(day)).days
                                            / 365.25
                                        )
                                        + r.midpoint(cq)
                                        - r.midpoint(pq)
                                    )
                                )
                            )
                            target = pending_shares if lagged else round(-current_delta / spot)
                            if lagged:
                                pending_shares = round(-current_delta / spot)
                            exec_cost += abs(target - shares) * (spot * 0.0001 + 0.005)
                            turnover += abs(target - shares) * spot
                            share_turnover += abs(target - shares)
                            shares = target
                        except ValueError:
                            pass  # Keep the prior observed hedge if IV/Greeks cannot be inverted.
                    if valid_stock:
                        prevspot = spot
                units[(signal, row.RIC, instrument)] = dict(
                    Entry=entry,
                    Dates=dates,
                    Mid=np.array(mid).T,
                    Spread=np.array(spread),
                    HedgeTurnover=np.array(hedge_turn),
                    Valid=np.array(complete),
                    LongExposureYears=np.array(long_exposure_years),
                    ShortExposureYears=np.array(short_exposure_years),
                    ShareTurnover=np.array(share_turnovers),
                    EntryPremium=p0,
                    EntrySpot=s0,
                    EntryDepth=[
                        min(
                            float(x.get("askSize") or 0)
                            for x in (
                                [quote(row.Call, row.Put, entry)[1]]
                                if side == "P"
                                else quote(row.Call, row.Put, entry)
                            )
                        ),
                        min(
                            float(x.get("bidSize") or 0)
                            for x in (
                                [quote(row.Call, row.Put, entry)[1]]
                                if side == "P"
                                else quote(row.Call, row.Put, entry)
                            )
                        ),
                    ],
                )
    return units
