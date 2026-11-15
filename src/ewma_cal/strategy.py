"""Frozen signals, explicit execution stresses, and reconciled position replay."""

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from .accounting import position_curve
from .cash import cash_interest_overlay
from .signals import calibration


class Strategy:
    def __init__(self, bundle, config):
        self.bundle = bundle
        self.config = config

    def history(self, source):
        return self.bundle.histories[source]

    @lru_cache(maxsize=5000)
    def details(self, specid, j, hedge, role="all"):
        r = self.history(specid).iloc[j]
        s = self.bundle.snapshots[r.SignalDate].set_index("RIC")
        qd = json.loads(r.Quantities)
        if role == "index_only":
            qd = {"DIA.P": qd["DIA.P"]}
        names = list(qd)
        q = np.array(list(qd.values()))
        instrument = self.bundle.source_specs[specid]["Env"]["INSTRUMENTS"]
        rows = [
            (
                k,
                self.bundle.units[hedge][
                    r.SignalDate, ric, instrument if ric == "DIA.P" else "straddle"
                ],
            )
            for k, ric in enumerate(names)
        ]
        var = self.bundle.source_specs[specid]["Book"] == "variance"
        contracts = np.array([1 if ric == "DIA.P" and instrument == "put" else 2 for ric in names])
        a = self.bundle.loc[r.EntryDate]
        planned = self.bundle.loc[r.PlannedExit]
        extra = (
            1.3 * sum((np.maximum(abs(q), 1) - abs(q)) * contracts) + 0.35 * len(q) * (r.H + 2)
            if var
            else 0.0
        )
        b, mid = position_curve(
            rows, q, self.bundle.days, a, planned, 0.0, extra, 0.35 * len(q) if var else 0.0
        )
        length = len(mid)
        spread = np.zeros(length)
        dollar_turn = np.zeros(length)
        share_turn = np.zeros(length)
        long_years = np.zeros(length)
        short_years = np.zeros(length)
        margin = 0.0
        depth = True
        signal_spread = 0.0
        vega = gamma = debit = 0.0
        notional = 0.0
        for k, u in rows:
            indices = [u["Dates"].index(d) for d in self.bundle.days[a : b + 1]]
            weight = abs(q[k])
            positive = q[k] > 0
            spread += weight * u["Spread"][indices]
            dollar_turn += weight * u["HedgeTurnover"][indices]
            share_turn += weight * u["ShareTurnover"][indices]
            long_years += (
                weight * u["LongExposureYears" if positive else "ShortExposureYears"][indices]
            )
            short_years += (
                weight * u["ShortExposureYears" if positive else "LongExposureYears"][indices]
            )
            debit += (
                max(q[k], 0)
                * u["EntryPremium"]
                * self.bundle.calendar["Rates"][r.EntryDate]
                * (pd.Timestamp(self.bundle.days[b]) - pd.Timestamp(r.EntryDate)).days
                / 365.25
            )
            notional += weight * 100 * u["EntrySpot"]
            margin += (
                weight * u["EntryPremium"]
                + max(-q[k], 0) * 0.2 * 100 * u["EntrySpot"]
                + 0.25 * weight * u["HedgeTurnover"][0]
            )
            depth = depth and weight <= u["EntryDepth"][0 if positive else 1]
            hs = s.loc[names[k], "HalfSpread"]
            factor = 0.5 if names[k] == "DIA.P" and instrument == "put" else 1.0
            if factor == 0.5:
                hs = 50 * (
                    float(s.loc[names[k], "Pair"][5]["ask"])
                    - float(s.loc[names[k], "Pair"][5]["bid"])
                )
            signal_spread += weight * 2 * hs
            vega += q[k] * s.loc[names[k], "Vega"] * factor
            gamma += q[k] * s.loc[names[k], "Gamma"] * factor
        return dict(
            R=r,
            A=a,
            B=b,
            Mid=mid,
            Spread=spread,
            DollarTurn=dollar_turn,
            ShareTurn=share_turn,
            LongYears=long_years,
            ShortYears=short_years,
            Q=q,
            Contracts=contracts,
            Variance=var,
            MinimumExtra=extra,
            Margin=margin,
            FullDepth=depth,
            SignalSpread=signal_spread,
            NetVega=vega,
            NetGamma=gamma,
            DebitFunding=debit,
            EntryNotional=notional,
        )

    def curve(
        self,
        d,
        spread=0.0,
        scale=1.0,
        commission=0.65,
        stock_bps=1.0,
        share_fee=0.005,
        borrow=0.01,
        funding_spread=0.03,
        handling=0.35,
        role="all",
    ):
        if role != "all":
            raise ValueError("Role ablations use independent index-only accounting")
        q = abs(d["Q"])
        newq = q * scale
        delay = np.maximum(
            np.arange(len(d["Mid"])) - (self.bundle.loc[d["R"].PlannedExit] - d["A"]), 0
        )
        old_extra = d["MinimumExtra"] + (0.35 * len(q) * delay if d["Variance"] else 0.0)
        value = scale * (d["Mid"] + old_extra)
        if d["Variance"]:
            extra = 2 * 0.65 * sum((np.maximum(newq, 1) - newq) * d["Contracts"]) + handling * len(
                q
            ) * (d["R"].H + 2)
            extra += handling * len(q) * delay
            value -= extra
        value -= spread * scale * d["Spread"]
        commission_units = 2 * sum(
            (np.maximum(newq, 1) if d["Variance"] else newq) * d["Contracts"]
        )
        value -= (commission - 0.65) * commission_units
        value -= scale * (
            (stock_bps - 1) / 10000 * d["DollarTurn"]
            + (share_fee - 0.005) * d["ShareTurn"]
            + (borrow - 0.01) * d["ShortYears"]
            + (funding_spread - 0.03) * d["LongYears"]
        )
        return value

    @lru_cache(maxsize=8192)
    def selected(
        self, specid, decay, gate, blend, hedge="post_quote", cost_recalibrate=False, **cost
    ):
        h = self.history(specid)
        rows = []
        if h.empty:
            return rows
        training = h.copy()
        if cost_recalibrate:
            training["NetUSD"] = [
                self.curve(self.details(specid, j, hedge), **cost)[-1] for j in range(len(h))
            ]
            training["NetForecast"] = h.NetForecast - float(cost.get("spread", 0)) * np.array(
                [self.details(specid, j, hedge)["SignalSpread"] for j in range(len(h))]
            )
        for j, r in h.iterrows():
            if not np.isfinite([r.ForecastRisk, r.NetForecast]).all() or r.ForecastRisk <= 0:
                raise ValueError("Invalid current forecast or risk")
            fit = calibration(training, r.SignalDate, decay)
            if fit is None:
                continue
            mu, residual, last = fit
            if blend is None:
                pred = mu * r.ForecastRisk
                enter = mu > gate
            else:
                current_forecast = (
                    float(training.NetForecast.iloc[j]) if cost_recalibrate else r.NetForecast
                )
                pred = blend * mu * r.ForecastRisk + (1 - blend) * (
                    current_forecast + residual * r.ForecastRisk
                )
                enter = pred > gate
            if enter and np.isfinite(pred):
                rows.append((j, float(pred), mu, residual, last))
        return rows

    def simulate(
        self,
        vega_source="baseline_vega",
        variance_source="baseline_variance",
        vega_decay=0.8,
        vega_gate=None,
        variance_decay=0.94,
        variance_blend=0.5,
        hedge=None,
        fill_probability=1.0,
        seed=0,
        strict_depth=False,
        cost_recalibrate=False,
        no_calibration=False,
        active_books="both",
        role="all",
        **cost,
    ):
        if vega_gate is None:
            vega_gate = self.config.vega_gate
        if hedge is None:
            hedge = self.config.hedge_timing
        pnl = np.zeros(len(self.bundle.days))
        records = []
        exposure = np.zeros(len(self.bundle.days))
        vega_path = np.zeros(len(self.bundle.days))
        gamma_path = np.zeros(len(self.bundle.days))
        active = np.zeros(len(self.bundle.days))
        rng = np.random.default_rng(seed)
        for specid, decay, gate, blend in [
            (vega_source, vega_decay, vega_gate, None),
            (variance_source, variance_decay, 0.0, variance_blend),
        ]:
            if active_books != "both" and self.bundle.source_specs[specid]["Book"] != active_books:
                continue
            choices = (
                self.selected(
                    specid,
                    decay,
                    gate,
                    blend,
                    hedge if cost_recalibrate else "post_quote",
                    cost_recalibrate,
                    **cost if cost_recalibrate else {},
                )
                if not no_calibration
                else [
                    (j, float(r.NetForecast), np.nan, np.nan, "unused")
                    for j, r in self.history(specid).iterrows()
                ]
            )
            for j, pred, mu, residual, last in choices:
                d = self.details(specid, j, hedge, role)
                if strict_depth and (not d["FullDepth"]):
                    continue
                if fill_probability < 1 and rng.random() >= fill_probability:
                    continue
                c = self.curve(d, **cost)
                a = d["A"]
                b = d["B"]
                pnl[a : b + 1] += c
                pnl[b + 1 :] += c[-1]
                scale = float(cost.get("scale", 1))
                exposure[a : b + 1] += d["Margin"] * scale
                vega_path[a : b + 1] += d["NetVega"] * scale
                gamma_path[a : b + 1] += d["NetGamma"] * scale
                active[a : b + 1] += 1
                r = d["R"]
                option_fees = (
                    2
                    * float(cost.get("commission", 0.65))
                    * sum(
                        (
                            np.maximum(abs(d["Q"]) * scale, 1)
                            if d["Variance"]
                            else abs(d["Q"]) * scale
                        )
                        * d["Contracts"]
                    )
                )
                stock_cost = scale * (
                    float(cost.get("stock_bps", 1)) / 10000 * d["DollarTurn"][-1]
                    + float(cost.get("share_fee", 0.005)) * d["ShareTurn"][-1]
                )
                stock_rf = scale * np.sum(
                    np.diff(d["LongYears"])
                    * np.array(
                        [self.bundle.calendar["Rates"][self.bundle.days[k]] for k in range(a, b)]
                    )
                )
                records.append(
                    dict(
                        r,
                        Book=self.bundle.source_specs[specid]["Book"],
                        Role=role,
                        Source=specid,
                        EntryIndex=a,
                        ExitIndex=b,
                        NetUSD=float(c[-1]),
                        ForecastMidUSD=pred,
                        GateMean=mu,
                        GateResidual=residual,
                        LastTrainingExit=last,
                        MissingDaily=int(np.isnan(c).sum()),
                        Margin=scale * d["Margin"],
                        OptionSpreadCostUSD=float(cost.get("spread", 0) * scale * d["Spread"][-1]),
                        DollarHedgeTurnover=float(scale * d["DollarTurn"][-1]),
                        NetVega=float(scale * d["NetVega"]),
                        NetGamma=float(scale * d["NetGamma"]),
                        FullDepth=d["FullDepth"],
                        OptionCommissionUSD=float(option_fees),
                        StockExecutionCostUSD=float(stock_cost),
                        StockBorrowUSD=float(
                            scale * cost.get("borrow", 0.01) * d["ShortYears"][-1]
                        ),
                        StockFundingSpreadUSD=float(
                            scale * cost.get("funding_spread", 0.03) * d["LongYears"][-1]
                        ),
                        StockRFFundingUSD=float(stock_rf),
                        OptionDebitFundingUSD=float(scale * d["DebitFunding"]),
                        HandlingCostUSD=float(
                            cost.get("handling", 0.35)
                            * len(d["Q"])
                            * (r.H + 2 + max(b - self.bundle.loc[r.PlannedExit], 0))
                        )
                        if d["Variance"]
                        else 0.0,
                        GrossEntryNotionalProxy=float(scale * d["EntryNotional"]),
                        Contribution=c,
                    )
                )
        return dict(
            PnL=pnl,
            Trades=records,
            Margin=exposure,
            Active=active,
            NetVega=vega_path,
            NetGamma=gamma_path,
        )

    def period_data(self, result, start, end, cash=False, cash_fraction=1.0):
        begin = int(np.flatnonzero(self.bundle.dates >= pd.Timestamp(start))[0])
        finish = int(np.flatnonzero(self.bundle.dates <= pd.Timestamp(end))[-1])
        anchor = begin
        ix = np.arange(anchor, finish + 1)
        initial = result["PnL"][anchor] if anchor else 0.0
        base = self.config.capital + result["PnL"][ix] - initial
        flows = {}
        for r in result["Trades"]:
            a = r["EntryIndex"]
            b = r["ExitIndex"]
            if b <= anchor or b > finish:
                continue
            net = r["NetUSD"]
            if a <= anchor < b:
                net -= r["Contribution"][anchor - a]
            flows[self.bundle.days[b]] = flows.get(self.bundle.days[b], 0.0) + net
        ledger = cash_interest_overlay(
            base,
            self.bundle.dates[ix],
            {d: v * cash_fraction for d, v in self.bundle.calendar["Rates"].items()},
            flows,
            self.config.capital,
        )
        rates = pd.Series(self.bundle.calendar["Rates"])
        rates.index = pd.to_datetime(rates.index)
        rf = rates.reindex(self.bundle.dates[ix]).to_numpy()
        dt = np.diff(self.bundle.dates[ix].asi8) / (86400000000000.0 * 365.25)
        rfreturn = np.r_[0.0, rf[:-1] * dt]
        return (ix, ledger, rfreturn)
