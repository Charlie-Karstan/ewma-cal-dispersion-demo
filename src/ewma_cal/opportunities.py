"""Exact-position shadow opportunities for the two frozen EWMA-CAL sleeves."""

import itertools
import json

import numpy as np
import pandas as pd

from .accounting import position_curve
from .pricing import option_expectation, state


def vega_quantities(s, chosen, iv, expected, C, policy, capital=1000000.0):
    result = []
    stockv = iv["Vega"][chosen].sum()
    stockg = (iv["Gamma"][chosen] * iv["Beta"][chosen] ** 2).sum()
    for ns in [1, 2, 3, 4]:
        for target in [stockv / iv["Vega"][0], stockg / iv["Gamma"][0]]:
            nd = max(1, int(np.floor(ns * target + 0.5)))
            q = np.zeros(31)
            q[chosen] = ns
            q[0] = -nd
            ve = abs(nd * iv["Vega"][0] / (ns * stockv) - 1)
            ge = abs(nd * iv["Gamma"][0] / (ns * stockg) - 1) if stockg > 0 else np.inf
            if policy == "vega" and (ve > 0.2 or ge > 1.0):
                continue
            if policy == "gamma" and (ge > 0.2 or ve > 1.0):
                continue
            if policy == "dual" and max(ve, ge) > 0.35:
                continue
            gross = 100 * sum(abs(q) * s.Spot.to_numpy())
            if gross > 2 * capital:
                continue
            result.append((q, ve, ge, gross))
    return result


def generate_vega_opportunities(bundle):
    snap = bundle.snapshots
    covs = {"EWMA_0.94": bundle.covariances}
    units = bundle.units["post_quote"]
    days = bundle.days
    loc = bundle.loc
    actions = bundle.actions
    divs = bundle.dividends
    settings = {
        "MODELS": "EWMA_0.94",
        "IV_DECAYS": ".3",
        "HORIZONS": "5",
        "INSTRUMENTS": "put",
        "POLICIES": "vega",
        "FILLS": "0",
        "BASKET_SIZES": "5,10,15,30",
        "MAX_TRACKING": ".75",
        "MAX_CONCENTRATION": ".4",
        "MIN_VARIANCE_COVERAGE": ".85",
    }
    fee = 0.65
    for model, states in covs.items():
        if model not in settings.get("MODELS", "EWMA_0.94,EWMA_0.97,DCC_252").split(","):
            continue
        state_cache = {}
        pred_cache = {}
        candidate_cache = {}
        for decay_iv in map(float, settings.get("IV_DECAYS", "0,.10,.30,1.0").split(",")):
            for H, instrument, policy, fill in itertools.product(
                list(map(int, settings.get("HORIZONS", "5,10,20").split(","))),
                settings.get("INSTRUMENTS", "put").split(","),
                settings.get("POLICIES", "vega,gamma,dual").split(","),
                list(map(float, settings.get("FILLS", "0").split(","))),
            ):
                busy = -1
                records = []
                prem_state = {}
                for signal, s in snap.items():
                    i = loc[signal]
                    if signal not in states or i + 1 + H >= len(days):
                        continue
                    if signal < "2022-01-01" and days[i + 1 + H] >= "2022-01-01":
                        continue
                    c = states[signal]
                    statekey = (signal, H)
                    if statekey not in state_cache:
                        state_cache[statekey] = state(s, c, model, H)
                    C0, CH, CE = state_cache[statekey]
                    pvar0 = 252 * np.diag(C0)
                    pvarend = 252 * np.diag(CE)
                    means = []
                    for ric, gap in zip(s.RIC, s.IV.to_numpy() ** 2 - pvar0):
                        old = prem_state.get(ric, gap)
                        means.append(old)
                        prem_state[ric] = 0.97 * old + 0.03 * gap
                    if signal < "2018-01-01" or i + 1 <= busy:
                        continue
                    entry = days[i + 1]
                    exitday = days[i + 1 + H]
                    tau = (
                        pd.to_datetime(s.Expiry) - pd.Timestamp(signal)
                    ).dt.days.to_numpy() / 365.25
                    u1 = (pd.Timestamp(entry) - pd.Timestamp(signal)).days / 365.25
                    ue = (pd.Timestamp(exitday) - pd.Timestamp(signal)).days / 365.25
                    if min(tau - ue) < 1 / 365.25:
                        continue
                    predkey = (signal, decay_iv, H, instrument)
                    if predkey not in pred_cache:
                        qtarget0 = np.maximum(pvar0 + means, 0.005**2)
                        qtargetend = np.maximum(pvarend + means, 0.005**2)
                        iv2 = s.IV.to_numpy() ** 2
                        iv_entry = np.sqrt(
                            np.maximum(
                                np.exp(-decay_iv) * iv2 + (1 - np.exp(-decay_iv)) * qtarget0,
                                0.005**2,
                            )
                        )
                        iv_exit = np.sqrt(
                            np.maximum(
                                np.exp(-decay_iv * (H + 1)) * iv2
                                + (1 - np.exp(-decay_iv * (H + 1))) * qtargetend,
                                0.005**2,
                            )
                        )
                        future_entry = option_expectation(
                            s, np.diag(C0), iv_entry, u1, tau - u1, instrument
                        )
                        future_exit = option_expectation(
                            s, np.diag(C0 + CH), iv_exit, ue, tau - ue, instrument
                        )
                        expected = future_exit - future_entry
                        pred_cache[predkey] = expected
                    expected = pred_cache[predkey]
                    ve = s.Vega.to_numpy().copy()
                    ga = s.Gamma.to_numpy().copy()
                    de = s.Delta.to_numpy().copy()
                    hs = s.HalfSpread.to_numpy().copy()
                    pr = s.Premium.to_numpy().copy()
                    contracts = np.full(31, 2.0)
                    if instrument == "put":
                        ve[0] *= 0.5
                        ga[0] *= 0.5
                        forward_disc = s.iloc[0].Spot * np.exp(-s.iloc[0].QYield * tau[0])
                        de[0] = 0.5 * (de[0] - 100 * forward_disc)
                        pair = s.iloc[0].Pair
                        hs[0] = 50 * (float(pair[5]["ask"]) - float(pair[5]["bid"]))
                        pr[0] = 50 * (float(pair[5]["ask"]) + float(pair[5]["bid"]))
                        contracts[0] = 1
                    beta = C0[:, 0] / max(C0[0, 0], 1e-12)
                    iv = dict(Vega=ve, Gamma=ga, Beta=beta)
                    w = s.Weight.to_numpy()[1:]
                    Cstock = C0[1:, 1:]
                    basevol = np.sqrt(max(w @ Cstock @ w, 1e-12))
                    candidates = []
                    candidatekey = (signal, decay_iv, H, instrument, policy)
                    if candidatekey not in candidate_cache:
                        for direction in {"both": [1, -1], "long": [1], "reverse": [-1]}[
                            settings.get("DIRECTION", "both")
                        ]:
                            orders = [
                                np.argsort(-s.Weight.to_numpy()[1:]) + 1,
                                np.argsort((2 * hs[1:] + 2 * fee * contracts[1:]) / ve[1:]) + 1,
                                np.argsort(-direction * expected[1:] / ve[1:]) + 1,
                            ]
                            seen = set()
                            for size, order in itertools.product(
                                list(
                                    map(int, settings.get("BASKET_SIZES", "5,10,15,30").split(","))
                                ),
                                orders,
                            ):
                                chosen = np.sort(order[:size])
                                key = tuple(chosen)
                                if key in seen:
                                    continue
                                seen.add(key)
                                a = np.zeros(30)
                                a[chosen - 1] = ve[chosen] / sum(ve[chosen])
                                tracking = np.sqrt(max((a - w) @ Cstock @ (a - w), 0)) / basevol
                                concentration = max(a)
                                if tracking > float(
                                    settings.get("MAX_TRACKING", ".75")
                                ) or concentration > float(settings.get("MAX_CONCENTRATION", ".4")):
                                    continue
                                for q, ve_error, ga_error, gross in vega_quantities(
                                    s, chosen, iv, expected, C0, policy
                                ):
                                    q *= direction
                                    if any(
                                        (
                                            ann <= signal and entry <= event <= exitday
                                            for j, ric in enumerate(s.RIC)
                                            if q[j] != 0
                                            for ann, event in actions.get(ric, [])
                                        )
                                    ):
                                        continue
                                    if settings.get("ALLOW_DIVIDENDS", "0") != "1" and any(
                                        (
                                            q[j] < 0
                                            and (instrument != "put" or j != 0)
                                            and any(
                                                (
                                                    ann <= signal and entry <= ex <= exitday
                                                    for ann, ex in divs.get(
                                                        ric.replace(".", "_").replace("^", "_"), []
                                                    )
                                                )
                                            )
                                            for j, ric in enumerate(s.RIC)
                                        )
                                    ):
                                        continue
                                    expected_option_cost = fill * 2 * sum(
                                        abs(q) * hs
                                    ) + 2 * fee * sum(abs(q) * contracts)
                                    holding_years = (
                                        pd.Timestamp(exitday) - pd.Timestamp(entry)
                                    ).days / 365.25
                                    turnover = 2 * abs(q * de) + abs(q) * ga * np.sqrt(
                                        np.maximum(pvar0, 0) / 252
                                    ) * H * np.sqrt(2 / np.pi)
                                    hedge_cost = sum(
                                        turnover * (0.0001 + 0.005 / s.Spot.to_numpy())
                                    )
                                    hd = -q * de
                                    carry = holding_years * (
                                        sum(np.maximum(hd, 0)) * (s.Rate.iloc[0] + 0.03)
                                        + sum(np.maximum(-hd, 0)) * 0.01
                                        + sum(np.maximum(q, 0) * pr) * s.Rate.iloc[0]
                                    )
                                    net = float(
                                        q @ expected - expected_option_cost - hedge_cost - carry
                                    )
                                    gamma = q * ga
                                    variance = 0.5 * H * (
                                        gamma @ (C0 * C0) @ gamma
                                    ) + H * 0.01**2 * (
                                        0.2 * sum((q * ve) ** 2) + 0.8 * sum(q * ve) ** 2
                                    )
                                    risk = np.sqrt(max(variance, 1))
                                    score = net / risk
                                    if net > 0 or settings.get("GATE", "positive") == "always":
                                        candidates.append(
                                            dict(
                                                Q=q,
                                                Net=net,
                                                Risk=risk,
                                                Score=score,
                                                GrossForecast=net + fill * 2 * sum(abs(q) * hs),
                                                SpreadForecast=2 * sum(abs(q) * hs),
                                                Tracking=tracking,
                                                Concentration=concentration,
                                                VegaError=ve_error,
                                                GammaError=ga_error,
                                                Gross=gross,
                                                Names=[s.RIC.iloc[j] for j in chosen],
                                            )
                                        )
                        candidate_cache[candidatekey] = candidates
                    else:
                        candidates = []
                        for item in candidate_cache[candidatekey]:
                            newnet = item["GrossForecast"] - fill * item["SpreadForecast"]
                            if newnet > 0 or settings.get("GATE", "positive") == "always":
                                candidates.append(
                                    dict(item, Net=newnet, Score=newnet / item["Risk"])
                                )
                    if not candidates:
                        continue
                    winner = None
                    rows = None
                    for proposal in sorted(candidates, key=lambda a: a["Score"], reverse=True):
                        q = proposal["Q"]
                        active = np.flatnonzero(q)
                        trial_rows = []
                        available = True
                        margin = 0.0
                        for j in active:
                            unit = units.get(
                                (signal, s.RIC.iloc[j], instrument if j == 0 else "straddle")
                            )
                            if unit is None:
                                available = False
                                break
                            selected = (
                                s.Pair.iloc[j][5:6]
                                if j == 0 and instrument == "put"
                                else s.Pair.iloc[j][4:6]
                            )
                            quote_capacity = min(
                                (
                                    max(
                                        float(z.get("askSize" if q[j] > 0 else "bidSize") or 0),
                                        0.1 * float(z.get("volume") or 0),
                                    )
                                    for z in selected
                                )
                            )
                            if fill == 1:
                                quote_capacity = unit["EntryDepth"][0 if q[j] > 0 else 1]
                            if abs(q[j]) > (
                                max(0, quote_capacity) if fill == 1 else max(1, quote_capacity)
                            ):
                                available = False
                                break
                            margin += (
                                abs(q[j]) * unit["EntryPremium"]
                                + max(-q[j], 0) * 0.2 * 100 * unit["EntrySpot"]
                                + 0.25 * abs(q[j]) * unit["HedgeTurnover"][0]
                            )
                            trial_rows.append((j, unit))
                        if available and margin <= 700000:
                            winner = proposal
                            rows = trial_rows
                            break
                    if winner is None:
                        continue
                    q = winner["Q"]
                    planned = i + 1 + H
                    busy, curve = position_curve(
                        rows,
                        q,
                        days,
                        i + 1,
                        planned,
                        fill,
                        winner.get("MinimumCostExtra", 0.0),
                        0.35 * sum(q != 0) if policy == "variance" else 0.0,
                    )
                    rec = dict(
                        SignalDate=signal,
                        EntryDate=entry,
                        ExitDate=days[busy],
                        PlannedExit=exitday,
                        ExitDelay=busy - planned,
                        ActualHold=busy - i - 1,
                        NetForecast=winner["Net"],
                        ForecastRisk=winner["Risk"],
                        Score=winner["Score"],
                        H=H,
                        Stocks=len(winner["Names"]),
                        Names=",".join(winner["Names"]),
                        WeightCoverage=float(s.loc[s.RIC.isin(winner["Names"]), "Weight"].sum()),
                        Quantities=json.dumps(dict(zip(s.RIC.iloc[active], q[active].tolist()))),
                        Tracking=winner["Tracking"],
                        Concentration=winner["Concentration"],
                        VegaError=winner["VegaError"],
                        GammaError=winner["GammaError"],
                        NetUSD=float(curve[-1]),
                        MissingDaily=int(np.isnan(curve).sum()),
                    )
                    records.append(rec)
                return pd.DataFrame(records)


def variance_quantities(s, chosen, iv, expected, C, policy, capital=1000000.0):
    result = []
    for nd in [4, 8, 16, 32, 64]:
        q = np.zeros(31)
        q[0] = -nd
        index_variance_units = nd * iv["Vega"][0] / (2 * iv["Sigma"][0])
        q[chosen] = (
            index_variance_units
            * iv["Weights"][chosen] ** 2
            / (iv["Vega"][chosen] / (2 * iv["Sigma"][chosen]))
        )
        stockv = sum(q[chosen] * iv["Vega"][chosen])
        stockg = sum(q[chosen] * iv["Gamma"][chosen] * iv["Beta"][chosen] ** 2)
        ve = abs(nd * iv["Vega"][0] / stockv - 1)
        ge = abs(nd * iv["Gamma"][0] / stockg - 1)
        gross = 100 * sum(abs(q) * s.Spot.to_numpy())
        if gross <= 2 * capital:
            result.append((q, ve, ge, gross))
    return result


def generate_variance_opportunities(bundle):
    snap = bundle.snapshots
    covs = {"EWMA_0.94": bundle.covariances}
    units = bundle.units["post_quote"]
    days = bundle.days
    loc = bundle.loc
    actions = bundle.actions
    divs = bundle.dividends
    settings = {
        "MODELS": "EWMA_0.94",
        "IV_DECAYS": "0",
        "HORIZONS": "5",
        "INSTRUMENTS": "put",
        "POLICIES": "vega",
        "FILLS": "0",
        "BASKET_SIZES": "5,10,15,30",
        "MAX_TRACKING": ".75",
        "MAX_CONCENTRATION": ".4",
        "MIN_VARIANCE_COVERAGE": ".85",
    }
    fee = 0.65
    for model, states in covs.items():
        if model not in settings.get("MODELS", "EWMA_0.94,EWMA_0.97,DCC_252").split(","):
            continue
        state_cache = {}
        pred_cache = {}
        candidate_cache = {}
        for decay_iv in map(float, settings.get("IV_DECAYS", "0,.10,.30,1.0").split(",")):
            for H, instrument, policy, fill in itertools.product(
                list(map(int, settings.get("HORIZONS", "5,10,20").split(","))),
                settings.get("INSTRUMENTS", "put").split(","),
                ["variance"],
                list(map(float, settings.get("FILLS", "0").split(","))),
            ):
                busy = -1
                records = []
                prem_state = {}
                for signal, s in snap.items():
                    i = loc[signal]
                    if signal not in states or i + 1 + H >= len(days):
                        continue
                    if signal < "2022-01-01" and days[i + 1 + H] >= "2022-01-01":
                        continue
                    c = states[signal]
                    statekey = (signal, H)
                    if statekey not in state_cache:
                        state_cache[statekey] = state(s, c, model, H)
                    C0, CH, CE = state_cache[statekey]
                    pvar0 = 252 * np.diag(C0)
                    pvarend = 252 * np.diag(CE)
                    means = []
                    for ric, gap in zip(s.RIC, s.IV.to_numpy() ** 2 - pvar0):
                        old = prem_state.get(ric, gap)
                        means.append(old)
                        prem_state[ric] = 0.97 * old + 0.03 * gap
                    if signal < "2018-01-01" or i + 1 <= busy:
                        continue
                    entry = days[i + 1]
                    exitday = days[i + 1 + H]
                    tau = (
                        pd.to_datetime(s.Expiry) - pd.Timestamp(signal)
                    ).dt.days.to_numpy() / 365.25
                    u1 = (pd.Timestamp(entry) - pd.Timestamp(signal)).days / 365.25
                    ue = (pd.Timestamp(exitday) - pd.Timestamp(signal)).days / 365.25
                    if min(tau - ue) < 1 / 365.25:
                        continue
                    predkey = (signal, decay_iv, H, instrument)
                    if predkey not in pred_cache:
                        qtarget0 = np.maximum(pvar0 + means, 0.005**2)
                        qtargetend = np.maximum(pvarend + means, 0.005**2)
                        iv2 = s.IV.to_numpy() ** 2
                        iv_entry = np.sqrt(
                            np.maximum(
                                np.exp(-decay_iv) * iv2 + (1 - np.exp(-decay_iv)) * qtarget0,
                                0.005**2,
                            )
                        )
                        iv_exit = np.sqrt(
                            np.maximum(
                                np.exp(-decay_iv * (H + 1)) * iv2
                                + (1 - np.exp(-decay_iv * (H + 1))) * qtargetend,
                                0.005**2,
                            )
                        )
                        future_entry = option_expectation(
                            s, np.diag(C0), iv_entry, u1, tau - u1, instrument
                        )
                        future_exit = option_expectation(
                            s, np.diag(C0 + CH), iv_exit, ue, tau - ue, instrument
                        )
                        expected = future_exit - future_entry
                        pred_cache[predkey] = expected
                    expected = pred_cache[predkey]
                    ve = s.Vega.to_numpy().copy()
                    ga = s.Gamma.to_numpy().copy()
                    de = s.Delta.to_numpy().copy()
                    hs = s.HalfSpread.to_numpy().copy()
                    pr = s.Premium.to_numpy().copy()
                    contracts = np.full(31, 2.0)
                    if instrument == "put":
                        ve[0] *= 0.5
                        ga[0] *= 0.5
                        forward_disc = s.iloc[0].Spot * np.exp(-s.iloc[0].QYield * tau[0])
                        de[0] = 0.5 * (de[0] - 100 * forward_disc)
                        pair = s.iloc[0].Pair
                        hs[0] = 50 * (float(pair[5]["ask"]) - float(pair[5]["bid"]))
                        pr[0] = 50 * (float(pair[5]["ask"]) + float(pair[5]["bid"]))
                        contracts[0] = 1
                    beta = C0[:, 0] / max(C0[0, 0], 1e-12)
                    iv = dict(
                        Vega=ve,
                        Gamma=ga,
                        Beta=beta,
                        Sigma=s.IV.to_numpy(),
                        Weights=s.Weight.to_numpy(),
                    )
                    w = s.Weight.to_numpy()[1:]
                    Cstock = C0[1:, 1:]
                    basevol = np.sqrt(max(w @ Cstock @ w, 1e-12))
                    candidates = []
                    candidatekey = (signal, decay_iv, H, instrument, policy)
                    if candidatekey not in candidate_cache:
                        for direction in {"both": [1, -1], "long": [1], "reverse": [-1]}[
                            settings.get("DIRECTION", "both")
                        ]:
                            orders = [
                                np.argsort(-s.Weight.to_numpy()[1:]) + 1,
                                np.argsort((2 * hs[1:] + 2 * fee * contracts[1:]) / ve[1:]) + 1,
                                np.argsort(-direction * expected[1:] / ve[1:]) + 1,
                            ]
                            seen = set()
                            for size, order in itertools.product(
                                list(
                                    map(int, settings.get("BASKET_SIZES", "5,10,15,30").split(","))
                                ),
                                orders,
                            ):
                                chosen = np.sort(order[:size])
                                key = tuple(chosen)
                                if key in seen:
                                    continue
                                seen.add(key)
                                a = np.zeros(30)
                                a[chosen - 1] = ve[chosen] / sum(ve[chosen])
                                tracking = np.sqrt(max((a - w) @ Cstock @ (a - w), 0)) / basevol
                                concentration = max(a)
                                for q, ve_error, ga_error, gross in variance_quantities(
                                    s, chosen, iv, expected, C0, policy
                                ):
                                    q *= direction
                                    a = np.zeros(30)
                                    a[chosen - 1] = (
                                        abs(q[chosen])
                                        * ve[chosen]
                                        / sum(abs(q[chosen]) * ve[chosen])
                                    )
                                    tracking = np.sqrt(max((a - w) @ Cstock @ (a - w), 0)) / basevol
                                    concentration = max(a)
                                    variance_coverage = sum(
                                        w[chosen - 1] ** 2 * pvar0[chosen]
                                    ) / sum(w**2 * pvar0[1:])
                                    if (
                                        tracking > float(settings.get("MAX_TRACKING", ".75"))
                                        or concentration
                                        > float(settings.get("MAX_CONCENTRATION", ".4"))
                                        or variance_coverage
                                        < float(settings.get("MIN_VARIANCE_COVERAGE", ".85"))
                                    ):
                                        continue
                                    if any(
                                        (
                                            ann <= signal and entry <= event <= exitday
                                            for j, ric in enumerate(s.RIC)
                                            if q[j] != 0
                                            for ann, event in actions.get(ric, [])
                                        )
                                    ):
                                        continue
                                    if settings.get("ALLOW_DIVIDENDS", "0") != "1" and any(
                                        (
                                            q[j] < 0
                                            and (instrument != "put" or j != 0)
                                            and any(
                                                (
                                                    ann <= signal and entry <= ex <= exitday
                                                    for ann, ex in divs.get(
                                                        ric.replace(".", "_").replace("^", "_"), []
                                                    )
                                                )
                                            )
                                            for j, ric in enumerate(s.RIC)
                                        )
                                    ):
                                        continue
                                    minimum_commission_extra = (
                                        2
                                        * fee
                                        * sum(
                                            (np.maximum(abs(q), 1) - abs(q)) * contracts * (q != 0)
                                        )
                                    )
                                    hedge_minimum_extra = 0.35 * sum(q != 0) * (H + 2)
                                    expected_option_cost = (
                                        fill * 2 * sum(abs(q) * hs)
                                        + 2 * fee * sum(abs(q) * contracts)
                                        + minimum_commission_extra
                                        + hedge_minimum_extra
                                    )
                                    holding_years = (
                                        pd.Timestamp(exitday) - pd.Timestamp(entry)
                                    ).days / 365.25
                                    turnover = 2 * abs(q * de) + abs(q) * ga * np.sqrt(
                                        np.maximum(pvar0, 0) / 252
                                    ) * H * np.sqrt(2 / np.pi)
                                    hedge_cost = sum(
                                        turnover * (0.0001 + 0.005 / s.Spot.to_numpy())
                                    )
                                    hd = -q * de
                                    carry = holding_years * (
                                        sum(np.maximum(hd, 0)) * (s.Rate.iloc[0] + 0.03)
                                        + sum(np.maximum(-hd, 0)) * 0.01
                                        + sum(np.maximum(q, 0) * pr) * s.Rate.iloc[0]
                                    )
                                    net = float(
                                        q @ expected - expected_option_cost - hedge_cost - carry
                                    )
                                    gamma = q * ga
                                    variance = 0.5 * H * (
                                        gamma @ (C0 * C0) @ gamma
                                    ) + H * 0.01**2 * (
                                        0.2 * sum((q * ve) ** 2) + 0.8 * sum(q * ve) ** 2
                                    )
                                    risk = np.sqrt(max(variance, 1))
                                    score = net / risk
                                    if net > 0 or settings.get("GATE", "positive") == "always":
                                        candidates.append(
                                            dict(
                                                Q=q,
                                                Net=net,
                                                Risk=risk,
                                                Score=score,
                                                MinimumCostExtra=minimum_commission_extra
                                                + hedge_minimum_extra,
                                                VarianceCoverage=variance_coverage,
                                                GrossForecast=net + fill * 2 * sum(abs(q) * hs),
                                                SpreadForecast=2 * sum(abs(q) * hs),
                                                Tracking=tracking,
                                                Concentration=concentration,
                                                VegaError=ve_error,
                                                GammaError=ga_error,
                                                Gross=gross,
                                                Names=[s.RIC.iloc[j] for j in chosen],
                                            )
                                        )
                        candidate_cache[candidatekey] = candidates
                    else:
                        candidates = []
                        for item in candidate_cache[candidatekey]:
                            newnet = item["GrossForecast"] - fill * item["SpreadForecast"]
                            if newnet > 0 or settings.get("GATE", "positive") == "always":
                                candidates.append(
                                    dict(item, Net=newnet, Score=newnet / item["Risk"])
                                )
                    if not candidates:
                        continue
                    winner = None
                    rows = None
                    for proposal in sorted(candidates, key=lambda a: a["Score"], reverse=True):
                        q = proposal["Q"]
                        active = np.flatnonzero(q)
                        trial_rows = []
                        available = True
                        margin = 0.0
                        for j in active:
                            unit = units.get(
                                (signal, s.RIC.iloc[j], instrument if j == 0 else "straddle")
                            )
                            if unit is None:
                                available = False
                                break
                            selected = (
                                s.Pair.iloc[j][5:6]
                                if j == 0 and instrument == "put"
                                else s.Pair.iloc[j][4:6]
                            )
                            quote_capacity = min(
                                (
                                    max(
                                        float(z.get("askSize" if q[j] > 0 else "bidSize") or 0),
                                        0.1 * float(z.get("volume") or 0),
                                    )
                                    for z in selected
                                )
                            )
                            if fill == 1:
                                quote_capacity = unit["EntryDepth"][0 if q[j] > 0 else 1]
                            if abs(q[j]) > (
                                max(0, quote_capacity) if fill == 1 else max(1, quote_capacity)
                            ):
                                available = False
                                break
                            margin += (
                                abs(q[j]) * unit["EntryPremium"]
                                + max(-q[j], 0) * 0.2 * 100 * unit["EntrySpot"]
                                + 0.25 * abs(q[j]) * unit["HedgeTurnover"][0]
                            )
                            trial_rows.append((j, unit))
                        if available and margin <= 700000:
                            winner = proposal
                            rows = trial_rows
                            break
                    if winner is None:
                        continue
                    q = winner["Q"]
                    planned = i + 1 + H
                    busy, curve = position_curve(
                        rows,
                        q,
                        days,
                        i + 1,
                        planned,
                        fill,
                        winner.get("MinimumCostExtra", 0.0),
                        0.35 * sum(q != 0) if policy == "variance" else 0.0,
                    )
                    rec = dict(
                        SignalDate=signal,
                        EntryDate=entry,
                        ExitDate=days[busy],
                        PlannedExit=exitday,
                        ExitDelay=busy - planned,
                        ActualHold=busy - i - 1,
                        NetForecast=winner["Net"],
                        ForecastRisk=winner["Risk"],
                        Score=winner["Score"],
                        H=H,
                        Stocks=len(winner["Names"]),
                        Names=",".join(winner["Names"]),
                        VarianceCoverage=winner["VarianceCoverage"],
                        WeightCoverage=float(s.loc[s.RIC.isin(winner["Names"]), "Weight"].sum()),
                        Quantities=json.dumps(dict(zip(s.RIC.iloc[active], q[active].tolist()))),
                        Tracking=winner["Tracking"],
                        Concentration=winner["Concentration"],
                        VegaError=winner["VegaError"],
                        GammaError=winner["GammaError"],
                        NetUSD=float(curve[-1]),
                        MissingDaily=int(np.isnan(curve).sum()),
                    )
                    records.append(rec)
                return pd.DataFrame(records)
