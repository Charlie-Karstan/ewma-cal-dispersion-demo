"""Guard capacity clipping and the additional funding ledger independently."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ewma_cal.sizing import SizingConfig, replay_sized


class ExampleStrategy:
    def __init__(self):
        dates = pd.date_range("2020-01-02", periods=6)
        days = dates.strftime("%Y-%m-%d").tolist()
        self.bundle = SimpleNamespace(
            days=days,
            dates=dates,
            calendar={"Rates": dict.fromkeys(days, 0.05)},
            units={
                "strictly_lagged": {
                    ("2020-01-01", "DIA.P", "put"): {"EntryDepth": [100, 100]},
                    ("2020-01-02", "DIA.P", "put"): {"EntryDepth": [100, 100]},
                }
            },
        )
        self.config = SimpleNamespace(capital=2_000_000.0, hedge_timing="strictly_lagged")
        self.history_rows = pd.DataFrame({"SignalDate": ["2020-01-01", "2020-01-02"]})

    def history(self, source):
        return self.history_rows

    def details(self, source, j, hedge):
        return {
            "Q": np.array([1.0]),
            "Variance": False,
            "Contracts": np.array([1.0]),
            "Spread": np.arange(4.0),
        }

    def curve(self, detail, scale, spread):
        return scale * (np.arange(4.0) * 100 - spread * detail["Spread"])

    def simulate(self, hedge=None):
        trades = []
        for a in [0, 1]:
            trades.append(
                dict(
                    Source="vega",
                    EntryIndex=a,
                    ExitIndex=a + 3,
                    SignalDate=self.history_rows.SignalDate.iloc[a],
                    LastTrainingExit="2019-12-31",
                    EntryDate=self.bundle.days[a],
                    ForecastMidUSD=200.0,
                    NetForecast=150.0,
                    ForecastRisk=100.0,
                    OptionCommissionUSD=1.3,
                    HandlingCostUSD=0.0,
                    Quantities='{"DIA.P": 1}',
                    Margin=3_000_000.0,
                    NetVega=10.0,
                    NetGamma=20.0,
                    DollarHedgeTurnover=1000.0,
                    StockExecutionCostUSD=1.0,
                    StockBorrowUSD=2.0,
                    StockFundingSpreadUSD=3.0,
                    StockRFFundingUSD=4.0,
                    OptionDebitFundingUSD=5.0,
                    GrossEntryNotionalProxy=10000.0,
                )
            )
        return {"Trades": trades}


def test_capacity_is_clipped_at_entry_and_blocks_only_when_occupied():
    result = replay_sized(ExampleStrategy(), SizingConfig())
    assert len(result["Trades"]) == 1
    np.testing.assert_allclose(result["Margin"][:4], 4_000_000.0)
    assert result["Trades"][0]["PositionScale"] == pytest.approx(4 / 3)
    assert result["Trades"][0]["Quantities"] != result["Trades"][0]["BaseQuantities"]


def test_prior_exposure_and_rate_funding_fully_reconcile():
    strategy = ExampleStrategy()
    result = replay_sized(strategy, SizingConfig())
    expected = 4 * 2_000_000 * (0.05 + 0.03) / 365.25
    assert result["ExtraCollateralFunding"][0] == 0
    assert result["ExtraCollateralFunding"].sum() == pytest.approx(expected)
    assert result["Trades"][0]["ExtraCollateralFundingUSD"] == pytest.approx(expected)
    assert result["PnL"][-1] == pytest.approx(400 - expected)
    strategy.bundle.calendar["Rates"][strategy.bundle.days[-1]] = 10
    changed = replay_sized(strategy, SizingConfig())
    np.testing.assert_array_equal(changed["PnL"], result["PnL"])


def test_sizing_config_rejects_invalid_targets():
    for kwargs in [
        {"target_scale": 0},
        {"margin_cap_usd": -1},
        {"extra_funding_spread": -0.01},
        {"target_scale": np.nan},
    ]:
        with pytest.raises(ValueError):
            SizingConfig(**kwargs)


def test_frozen_model_hashes_are_intact():
    from pathlib import Path

    from ewma_cal.leverage import check_freeze

    root = Path(__file__).resolve().parents[1]
    assert check_freeze(root)["Name"] == "prof2_EWMA_CAL_size8_cap4m_funded"


def test_leverage_grid_reproduces_frozen_anchor():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    grid = pd.read_csv(root / "reports/leverage/leverage_metrics.csv")
    anchor = pd.read_csv(root / "reports/reference/metrics.csv")
    for period in ["Full", "Development", "OOS"]:
        for cash, portfolio in [(False, "Trading only"), (True, "With cash interest")]:
            a = grid[
                (grid.TargetScale == 8) & (grid.Period == period) & (grid.CashInterest == cash)
            ].iloc[0]
            b = anchor[
                anchor.Primary & (anchor.Period == period) & (anchor.Portfolio == portfolio)
            ].iloc[0]
            np.testing.assert_allclose(
                a[["NetPnL", "CAGR", "Sharpe", "MaxDrawdown"]].to_numpy(float),
                b[["NetPnL", "CAGR", "Sharpe", "MaxDrawdown"]].to_numpy(float),
                atol=1e-8,
            )
    assert (grid.MaxMarginProxy <= 4_000_000 + 1e-6).all()
    assert grid.TargetScale.nunique() == 10
