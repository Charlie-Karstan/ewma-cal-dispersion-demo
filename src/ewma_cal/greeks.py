"""Quoted straddle IV and Greeks using the frozen parity approximation."""

from datetime import date

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from . import market as research


def entry_pair(pair, call, put):
    if call is None or put is None:
        return None
    try:
        spot = float(call["underlyingPrice"])
        other = float(put["underlyingPrice"])
    except (KeyError, TypeError, ValueError):
        return None
    if (
        not np.isfinite([spot, other]).all()
        or spot <= 0
        or abs(spot - other) > max(0.05, 0.003 * spot)
    ):
        return None
    return (*pair[:4], call, put, spot)


def straddle(pair, entry, rate):
    """European parity/IV approximation for the quoted American straddle."""
    expiry, strike, call, put, spot = pair[2:]
    tau = (date.fromisoformat(expiry) - date.fromisoformat(entry)).days / 365.25
    if tau <= 0:
        raise ValueError("Expired contract")
    discount = np.exp(-rate * tau)
    premium = research.midpoint(call) + research.midpoint(put)
    forward = strike + (research.midpoint(call) - research.midpoint(put)) / discount
    if forward <= 0:
        raise ValueError("Invalid implied forward")
    log_moneyness = np.log(forward / strike)

    def coordinates(vol):
        width = vol * np.sqrt(tau)
        d1 = log_moneyness / width + width / 2
        return d1, d1 - width

    def residual(vol):
        d1, d2 = coordinates(vol)
        return discount * (forward * (2 * ndtr(d1) - 1) - strike * (2 * ndtr(d2) - 1)) - premium

    iv = brentq(residual, 0.001, 4.0)
    d1, _ = coordinates(iv)
    density = np.exp(-(d1**2) / 2) / np.sqrt(2 * np.pi)
    dollar_vega = 200 * discount * forward * density * np.sqrt(tau)
    half_spread = 50 * sum(float(q["ask"]) - float(q["bid"]) for q in (call, put))
    dollar_gamma = dollar_vega / (iv * tau)
    if min(dollar_vega, dollar_gamma) <= 1e-8:
        raise ValueError("Degenerate straddle risk")
    return dict(
        iv=iv,
        vega=dollar_vega,
        gamma=dollar_gamma,
        spread=half_spread,
        delta=100 * discount * forward * (2 * ndtr(d1) - 1),
    )
