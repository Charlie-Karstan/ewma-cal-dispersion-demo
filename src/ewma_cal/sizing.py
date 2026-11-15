"""Frozen capacity-clipped sizing and explicitly charged collateral financing."""

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SizingConfig:
    name: str = "prof2_EWMA_CAL_size8_cap4m_funded"
    version: str = "2.2.0"
    target_scale: float = 8.0
    margin_cap_usd: float = 4_000_000.0
    extra_funding_spread: float = 0.03

    def __post_init__(self):
        values = [self.target_scale, self.margin_cap_usd, self.extra_funding_spread]
        if not np.isfinite(values).all() or min(values[:2]) <= 0 or values[2] < 0:
            raise ValueError("Sizing requires positive scale/cap and nonnegative funding spread")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def as_dict(self):
        return asdict(self)


def replay_sized(strategy, sizing, hedge=None, spread=0.0, base=None):
    """Resize fixed causal signals at entry; never fit to leverage-test outcomes.

    The cap is an aggregate initial-capital margin proxy, not a broker margin
    guarantee. Quantities are option equivalents, including fractional units.
    Extra financing uses the prior session's exposure and Treasury rate. Its
    final interval settles with the existing exit cashflow, matching the frozen
    research funding convention. Calibration predictions precede that stress.
    """
    hedge = hedge or strategy.config.hedge_timing
    base = strategy.simulate(hedge=hedge) if base is None else base
    days, dates = strategy.bundle.days, strategy.bundle.dates
    result = {k: np.zeros(len(days)) for k in ["PnL", "Margin", "Active", "NetVega", "NetGamma"]}
    result["Trades"] = []
    groups = {}
    for original in base["Trades"]:
        if not original["LastTrainingExit"] < original["SignalDate"] < original["EntryDate"]:
            raise ValueError("Unmatured training label or noncausal entry")
        groups.setdefault(original["EntryIndex"], []).append(original)
    for a, group in sorted(groups.items()):
        available = max(sizing.margin_cap_usd - result["Margin"][a], 0.0)
        size = min(sizing.target_scale, available / sum(r["Margin"] for r in group))
        if size <= 1e-8:
            continue
        for original in group:
            r = copy.copy(original)
            history = strategy.history(r["Source"])
            j = int(history.index[history.SignalDate == r["SignalDate"]][0])
            detail = strategy.details(r["Source"], j, hedge)
            curve = strategy.curve(detail, scale=size, spread=spread)
            b = r["ExitIndex"]
            q = abs(detail["Q"]) * size
            fee = 1.3 * sum((np.maximum(q, 1) if detail["Variance"] else q) * detail["Contracts"])
            correction = size * (r["OptionCommissionUSD"] + r["HandlingCostUSD"]) - (
                fee + r["HandlingCostUSD"]
            )
            r["ForecastMidUSD"] = size * r["ForecastMidUSD"] + correction
            r["NetForecast"] = size * r["NetForecast"] + correction
            r["ForecastRisk"] *= size
            r["NetUSD"] = float(curve[-1])
            r["Contribution"] = curve
            r["PositionScale"] = size
            quantities = json.loads(r["Quantities"])
            r["AllLegsWithinDepth"] = all(
                abs(quantity) * size
                <= strategy.bundle.units[hedge][
                    r["SignalDate"], ric, "put" if ric == "DIA.P" else "straddle"
                ]["EntryDepth"][0 if quantity > 0 else 1]
                for ric, quantity in quantities.items()
            )
            r["BaseQuantities"] = r["Quantities"]
            r["Quantities"] = json.dumps(
                {ric: quantity * size for ric, quantity in quantities.items()}
            )
            r["OptionCommissionUSD"] = float(fee)
            r["OptionSpreadCostUSD"] = float(spread * size * detail["Spread"][-1])
            for key in [
                "DollarHedgeTurnover",
                "StockExecutionCostUSD",
                "StockBorrowUSD",
                "StockFundingSpreadUSD",
                "StockRFFundingUSD",
                "OptionDebitFundingUSD",
                "GrossEntryNotionalProxy",
            ]:
                r[key] *= size
            for key in ["Margin", "NetVega", "NetGamma"]:
                r[key] *= size
                result[key][a : b + 1] += r[key]
            result["Active"][a : b + 1] += 1
            result["Trades"].append(r)
    if result["Margin"].max() > sizing.margin_cap_usd + 1e-6:
        raise AssertionError("Aggregate margin-proxy cap breached")
    funding = np.zeros(len(days))
    for k in range(1, len(days)):
        dt = (dates[k] - dates[k - 1]).days / 365.25
        funding[k] = (
            max(result["Margin"][k - 1] - strategy.config.capital, 0)
            * (strategy.bundle.calendar["Rates"][days[k - 1]] + sizing.extra_funding_spread)
            * dt
        )
    allocated = np.zeros(len(days))
    for r in result["Trades"]:
        a, b = r["EntryIndex"], r["ExitIndex"]
        ix = np.arange(a + 1, min(b + 2, len(days)))
        costs = funding[ix] * np.divide(
            r["Margin"],
            result["Margin"][ix - 1],
            out=np.zeros(len(ix)),
            where=result["Margin"][ix - 1] > 0,
        )
        allocated[ix] += costs
        cumulative = np.zeros(b - a + 1)
        for k, cost in zip(ix, costs):
            cumulative[min(k - a, b - a) :] += cost
        r["Contribution"] = r["Contribution"] - cumulative
        r["NetUSD"] -= costs.sum()
        r["ExtraCollateralFundingUSD"] = float(costs.sum())
        r["MissingDaily"] = int(np.isnan(r["Contribution"]).sum())
        result["PnL"][a : b + 1] += r["Contribution"]
        result["PnL"][b + 1 :] += r["NetUSD"]
    np.testing.assert_allclose(allocated, funding, atol=1e-8)
    np.testing.assert_allclose(
        result["PnL"][-1], sum(r["NetUSD"] for r in result["Trades"]), atol=1e-8, equal_nan=True
    )
    result["ExtraCollateralFunding"] = funding
    return result
