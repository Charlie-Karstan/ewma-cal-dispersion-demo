"""Fit to finite labels that mature strictly before the signal date."""

import numpy as np


def calibration(history, signal, decay):
    if not np.isfinite(decay) or not 0 < decay <= 1:
        raise ValueError("EWMA decay must be finite and in (0, 1]")
    known = history[history.ExitDate < signal].sort_values("ExitDate", kind="stable").tail(40)
    if len(known) < 5:
        return None
    values = known[["NetUSD", "NetForecast", "ForecastRisk"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, 2] <= 0).any():
        return None  # A missing label cannot be interpreted as evidence to enter.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        outcomes = values[:, 0] / values[:, 2]
        forecasts = values[:, 1] / values[:, 2]
        if not np.isfinite(outcomes).all() or not np.isfinite(forecasts).all():
            return None
        weights = decay ** np.arange(len(known) - 1, -1, -1)
        mean = float(weights @ outcomes / weights.sum())
        residual = float(weights @ (outcomes - forecasts) / weights.sum())
    if not np.isfinite([mean, residual]).all():
        return None
    return mean, residual, str(known.ExitDate.max())
