"""Guard time ordering, execution accounting, and benchmark comparability."""

import numpy as np
import pandas as pd
import pytest

from ewma_cal.accounting import position_curve
from ewma_cal.cash import cash_interest_overlay
from ewma_cal.config import ModelConfig
from ewma_cal.covariance import estimate_covariances
from ewma_cal.metrics import forecast_metrics, summarize
from ewma_cal.reporting import read_benchmark
from ewma_cal.signals import calibration


def shadow_history():
    return pd.DataFrame(
        {
            "ExitDate": [f"2020-01-{k:02}" for k in range(1, 8)],
            "NetUSD": [2.0] * 7,
            "NetForecast": [1.0] * 7,
            "ForecastRisk": [2.0] * 7,
        }
    )


def test_calibration_excludes_unmatured_and_same_day_labels():
    h = shadow_history()
    before = calibration(h, "2020-01-06", 0.8)
    h.loc[h.ExitDate >= "2020-01-06", "NetUSD"] = 1e12
    assert calibration(h, "2020-01-06", 0.8) == before
    assert before[:2] == pytest.approx((1.0, 0.5))
    assert before[2] == "2020-01-05"
    assert calibration(h, "2020-01-05", 0.8) is None


def test_invalid_labels_never_become_an_entry():
    h = shadow_history()
    h.loc[2, "ForecastRisk"] = 0
    assert calibration(h, "2020-01-08", 0.8) is None
    h.loc[2, "ForecastRisk"] = 2
    h.loc[2, "NetUSD"] = np.nan
    assert calibration(h, "2020-01-08", 0.8) is None
    with pytest.raises(ValueError):
        calibration(h, "2020-01-08", 1.1)


def test_cash_credits_only_settled_balances_and_prior_rates():
    dates = pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-07"])
    ledger = cash_interest_overlay(
        [100, 1000, 110], dates, dict(zip(dates, [0.1, 0.2, 0.9])), {dates[-1]: 10}, 100
    )
    first_credit = 100 * 0.1 * 3 / 365.25
    second_credit = (100 + first_credit) * 0.2 / 365.25
    np.testing.assert_allclose(ledger.CashInterestCredit, [0, first_credit, second_credit])
    assert ledger.SettledCollateralCash.iloc[-1] == pytest.approx(
        110 + first_credit + second_credit
    )
    np.testing.assert_allclose(
        ledger.EquityWithInterest - ledger.EquityWithoutInterest, ledger.CumulativeCashInterest
    )


def test_missing_marks_preserve_entered_position_and_defer_exit():
    days = [f"2020-01-{k:02}" for k in range(1, 5)]
    unit = {"Dates": days, "Mid": np.array([[0, np.nan, np.nan, 8]] * 2), "Spread": np.zeros(4)}
    end, curve = position_curve([(0, unit)], np.array([1]), days, 0, 2, 0)
    assert end == 3 and curve[-1] == 8 and np.isnan(curve[1:3]).all()
    unit["Mid"][:, -1] = np.nan
    end, curve = position_curve([(0, unit)], np.array([1]), days, 0, 2, 0)
    assert end == 2 and np.isnan(curve[-1])


def test_metrics_do_not_silently_bridge_unknown_nav():
    dates = pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01"])
    row = summarize([100, 80, 110], dates, [0, 0, 0], [100, 90, 120])
    assert row["MaxDrawdown"] == pytest.approx(-0.2)
    assert row["TotalReturn"] == pytest.approx(0.1)
    assert row["Calmar"] == pytest.approx(row["CAGR"] / 0.2)
    row = summarize([100, np.nan, 110], dates, [0, 0, 0])
    assert row["MissingMarks"] == 1 and np.isnan(row["Sharpe"])
    assert row["TotalReturn"] == pytest.approx(0.1)


def test_forecast_error_units_are_explicit():
    result = forecast_metrics([2, 6], [4, 2], [2, 4])
    assert result["MSE"] == 10 and result["MAE"] == 3
    assert result["ASE"] == 1 and result["Bias"] == -1


def test_benchmark_requires_complete_aligned_prices(tmp_path):
    path = tmp_path / "spy.csv"
    pd.DataFrame({"Date": ["2020-01-01"], "AdjustedClose": [100]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing"):
        read_benchmark(path, pd.to_datetime(["2020-01-01", "2020-01-02"]))


def test_covariance_cannot_see_future_returns():
    dates = pd.date_range("2020-01-01", periods=31)
    rng = np.random.default_rng(4)
    returns = pd.DataFrame(rng.normal(0, 0.01, (31, 2)), index=dates, columns=["A", "B"])
    day = str(dates[25].date())
    snapshots = {day: pd.DataFrame({"RIC": ["DIA.P", "A", "B"], "IV": [0.2, 0.3, 0.4]})}
    a = estimate_covariances(snapshots, returns)[day]["Entry"]
    returns.loc[returns.index > day] = 1e9
    b = estimate_covariances(snapshots, returns)[day]["Entry"]
    np.testing.assert_array_equal(a, b)
    assert np.linalg.eigvalsh(a).min() > 0


def test_release_settings_are_frozen():
    config = ModelConfig()
    assert config.vega_gate == 0.15 and config.hedge_timing == "strictly_lagged"
    with pytest.raises(ValueError):
        ModelConfig(capital=-1)


def test_saved_results_reconcile_without_private_inputs():
    from pathlib import Path

    from ewma_cal.reporting import load_reference

    root = Path(__file__).resolve().parents[1]
    ref = load_reference(root)
    full = ref["equity"].query("Period == 'Full'")
    np.testing.assert_allclose(full.CashEquity - full.TradingEquity, full.CashInterest, atol=1e-8)
    assert ref["trades"].NetUSD.sum() == pytest.approx(full.TradingEquity.iloc[-1] - 2_000_000)
    assert (ref["trades"].LastTrainingExit < ref["trades"].SignalDate).all()
    for period, f in ref["equity"].groupby("Period"):
        assert f.Date.is_monotonic_increasing and f.Date.is_unique
        assert f[["TradingEquity", "CashEquity", "SP500Equity"]].iloc[0].eq(2_000_000).all()
        expected = (
            ref["metrics"]
            .query("Primary and Period == @period and Portfolio == 'Trading only'")
            .iloc[0]
        )
        assert expected.EndingEquity == pytest.approx(f.TradingEquity.iloc[-1])


def test_benchmark_download_keeps_adjusted_total_return(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    from ewma_cal.benchmark import download

    index = pd.to_datetime(["2020-01-01", "2020-01-02"])
    provider = pd.DataFrame({"Close": [100, 105], "Adj Close": [90, 95]}, index=index)
    calls = []

    def fake_download(ticker, **kwargs):
        calls.append((ticker, kwargs))
        return provider

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=fake_download))
    out = tmp_path / "benchmark.csv"
    result = download(out, "2020-01-01", "2020-01-03")
    assert result.AdjustedClose.tolist() == [90, 95]
    assert calls[0][0] == "SPY" and calls[0][1]["auto_adjust"] is False
    assert out.with_suffix(".source.json").exists()
