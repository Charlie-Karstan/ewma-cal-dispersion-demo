"""Read an existing local market archive; no network access or credentials."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")
OCC = re.compile(r"^[A-Z0-9]{1,6}\d{6}[CP]\d{8}$")


def trading_symbol(ric: str, signal: str) -> str:
    """Matches the validated historical mapping in the existing downloader."""
    if ric == "DIA.P":
        return "DIA"
    if ric == "RTX.N" and signal < "2020-04-03":
        return "UTX"
    if ric == "DD.N" and "2017-09-01" <= signal < "2019-06-03":
        return "DWDP"
    return ric.split(".", 1)[0].split("^", 1)[0]


def expiration(value) -> str | None:
    try:
        if isinstance(value, (int, float)):
            return (
                datetime.fromtimestamp(float(value), timezone.utc).astimezone(NY).date().isoformat()
            )
        return date.fromisoformat(str(value)[:10]).isoformat()
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def rows_from_payload(payload) -> list[dict]:
    if not isinstance(payload, dict) or payload.get("s") != "ok":
        return []
    symbols = payload.get("optionSymbol")
    if not isinstance(symbols, list):
        return []
    return [
        {
            k: (v[i] if isinstance(v, list) and len(v) == len(symbols) else None)
            for k, v in payload.items()
            if k != "s"
        }
        for i in range(len(symbols))
    ]


def quote_ok(q: dict) -> bool:
    try:
        b, a = float(q["bid"]), float(q["ask"])
        return all(map(math.isfinite, (b, a))) and 0 <= b <= a and a > 0
    except (ValueError, KeyError, TypeError):
        return False


def choose_pair(
    payload,
    signal: str,
    config,
    common_expiry: str | None = None,
    expected_root: str | None = None,
    *,
    ranked: bool = False,
):
    """Select closest ATM pair at the expiry nearest target DTE, not nearest expiry."""
    options = defaultdict(dict)
    for q in rows_from_payload(payload):
        if not quote_ok(q):
            continue
        side = str(q.get("side", "")).lower()
        ex = expiration(q.get("expiration"))
        if side not in ("call", "put") or ex is None:
            continue
        if common_expiry is not None and ex != common_expiry:
            continue
        dte = (date.fromisoformat(ex) - date.fromisoformat(signal)).days
        if not config.min_dte <= dte <= config.max_dte:
            continue
        try:
            strike, spot = float(q["strike"]), float(q["underlyingPrice"])
            symbol = str(q["optionSymbol"])
            if not all(map(math.isfinite, (strike, spot))) or min(strike, spot) <= 0:
                continue
            match = OCC.fullmatch(symbol)
            if match is None:
                continue
            if expected_root is not None and symbol[: match.end() - 15] != expected_root:
                continue
            # The historical quote download included only pairs whose signal-date
            # bidSize/askSize were both positive. Do not request uncached contracts.
            if min(float(q.get("bidSize") or 0), float(q.get("askSize") or 0)) <= 0:
                continue
        except (KeyError, ValueError, TypeError):
            continue
        options[(ex, strike)][side] = q
    pairs = []
    for (ex, strike), cp in options.items():
        if "call" not in cp or "put" not in cp:
            continue
        c, p = cp["call"], cp["put"]
        s, ps = float(c["underlyingPrice"]), float(p["underlyingPrice"])
        if abs(s - ps) > max(0.05, 0.003 * s):
            continue
        dte = (date.fromisoformat(ex) - date.fromisoformat(signal)).days
        pairs.append((abs(dte - config.target_dte), abs(strike - s) / s, ex, strike, c, p, s))
    pairs.sort(key=lambda x: (x[0], x[1], x[3]))
    return pairs if ranked else next(iter(pairs), None)


class Archive:
    def __init__(self, root: Path):
        self.raw = root / "raw/dow/options/full_history"
        if not (self.raw / "dataset_config.json").is_file():
            raise FileNotFoundError(f"Full option archive missing: {self.raw}")

    @lru_cache(maxsize=2048)
    def chain(self, day: str, symbol: str):
        return self._load(self.raw / "chains" / day / f"{symbol}.json")

    @lru_cache(maxsize=4096)
    def history(self, symbol: str) -> dict[str, dict]:
        payload = self._load(self.raw / "quotes" / f"{symbol}.json")
        history = {}
        for row in rows_from_payload(payload):
            try:
                stamp = float(row["updated"])
                day = datetime.fromtimestamp(stamp, timezone.utc).astimezone(NY).date().isoformat()
            except (TypeError, ValueError, KeyError, OverflowError, OSError):
                continue
            if quote_ok(row) and (day not in history or stamp > history[day][0]):
                history[day] = (stamp, row)
        return {d: q for d, (stamp, q) in history.items()}

    @staticmethod
    def _load(path: Path):
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None


def load_members(root):
    p = root / "processed/dow/dow_weights_daily.csv"
    w = pd.read_csv(p, dtype={"Date": str, "RIC": str})
    if not {"Date", "RIC", "Weight"} <= set(w.columns):
        raise ValueError("Point-in-time weights require Date, RIC, Weight")
    memberships = {}
    for day, group in w.groupby("Date"):
        if len(group) != 30 or group.RIC.nunique() != 30:
            raise ValueError(f"{day}: expected 30 Dow members")
        memberships[str(day)] = group.copy()
    return memberships


def midpoint(q):
    return (float(q["bid"]) + float(q["ask"])) / 2
