"""Execution-time exit deferral and observed-mark return accounting.
No quote interpolation, disappearance of entered trades, or interest on idle cash.
"""

import numpy as np
import pandas as pd


def position_curve(rows, q, days, start, planned_end, fill, minimum_extra=0.0, extra_daily=0.0):
    """Attempt the scheduled exit, then every session until all legs can be quoted.

    Each attempted date uses only that date's quotes. Entry never depends on
    future exit availability. Absent a joint exit before expiry, return NaN.
    The same position remains open, blocking later entries during any delay.
    """
    lookups = [{d: k for k, d in enumerate(unit["Dates"])} for _, unit in rows]
    last = min(max(lookup) for lookup in lookups)
    end = planned_end
    while end < len(days) and days[end] <= last:
        valid = True
        for (j, unit), lookup in zip(rows, lookups):
            k = lookup.get(days[end])
            side = 0 if q[j] > 0 else 1
            if k is None or not np.isfinite(unit["Mid"][side, k] - fill * unit["Spread"][k]):
                valid = False
                break
        if valid:
            break
        end += 1
    if end >= len(days) or days[end] > last:
        end = planned_end  # Unresolved exit is retained as an unknown outcome.
    curve = np.zeros(end - start + 1)
    for (j, unit), lookup in zip(rows, lookups):
        side = 0 if q[j] > 0 else 1
        vals = []
        for d in days[start : end + 1]:
            k = lookup.get(d)
            vals.append(
                unit["Mid"][side, k] - fill * unit["Spread"][k] if k is not None else np.nan
            )
        curve += abs(q[j]) * np.asarray(vals)
    curve -= minimum_extra
    # Handling minimum on additional hedge sessions incurred by delayed exits.
    curve -= extra_daily * np.maximum(np.arange(len(curve)) - (planned_end - start), 0)
    return end, curve


def monthly_statistics(values, dates):
    """First-session monthly marks; unavailable marks defer to first observed NAV.

    Deferrals are never filled with a zero return. Dates enter annualization,
    including the final stub, and the count of delayed valuations is disclosed.
    """
    dates = pd.DatetimeIndex(dates)
    values = np.asarray(values, float)
    scheduled = (
        pd.Series(range(len(values)), index=dates).groupby(dates.to_period("M")).first().to_numpy()
    )
    scheduled = np.unique(np.r_[scheduled, len(values) - 1])
    actual = []
    delays = []
    for n, k in enumerate(scheduled):
        limit = scheduled[n + 1] if n + 1 < len(scheduled) else len(values)
        observed = np.flatnonzero(np.isfinite(values[k:limit]))
        if not len(observed):
            return np.nan, len(scheduled) - 1, -1, -1
        chosen = k + observed[0]
        actual.append(chosen)
        delays.append(chosen - k)
    actual = np.unique(actual)
    marks = values[actual]
    r = marks[1:] / marks[:-1] - 1
    dt = np.diff(dates[actual].asi8) / (86400 * 1e9 * 365.25)
    mu = r.sum() / dt.sum() if len(r) else np.nan
    sigma = (
        np.sqrt(np.sum((r - mu * dt) ** 2) / dt.sum() * len(r) / (len(r) - 1))
        if len(r) > 1
        else np.nan
    )
    return (mu / sigma if sigma > 0 else np.nan), len(r), sum(x > 0 for x in delays), max(delays)
