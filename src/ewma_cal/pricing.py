"""Exact-strike option expectations and trade-aligned covariance projections."""

import numpy as np
from scipy.special import ndtr


def option_expectation(s, v, iv, u, rem, instrument):
    # Mean-zero total stock return under P; the hedge forecast has zero gross mean.
    f = (
        s.Spot.to_numpy()
        * np.exp(-s.QYield.to_numpy() * u)
        * np.exp((s.Rate.to_numpy() - s.QYield.to_numpy()) * rem)
    )
    k = s.Strike.to_numpy()
    variance = np.maximum(v + iv**2 * rem, 1e-12)
    sd = np.sqrt(variance)
    d1 = (np.log(f / k) + 0.5 * variance) / sd
    d2 = d1 - sd
    discount = np.exp(-s.Rate.to_numpy() * rem)
    values = discount * (f * (2 * ndtr(d1) - 1) - k * (2 * ndtr(d2) - 1)) * 100
    if instrument == "put":
        values[0] = discount[0] * (k[0] * ndtr(-d2[0]) - f[0] * ndtr(-d1[0])) * 100
    return values


def state(s, c, model, H):
    w = s.Weight.to_numpy()[1:]
    names = s.RIC.tolist()[1:]
    assert c["Names"] == names
    if "Projection" in c:
        entry = c["Projection"][0].astype(float)
        holding = np.sum(c["Projection"][1 : H + 1].astype(float), axis=0)
        end = c["Projection"][H].astype(float)
    else:
        entry = c["Entry"]
        holding = entry * H
        end = entry

    def extend(C):
        full = np.zeros((31, 31))
        full[1:, 1:] = C
        full[0, 0] = w @ C @ w
        full[0, 1:] = w @ C
        full[1:, 0] = C @ w
        return full

    return extend(entry), extend(holding), extend(end)
