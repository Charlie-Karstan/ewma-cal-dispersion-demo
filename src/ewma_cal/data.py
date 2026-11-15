"""Explicit, local-only inputs. Prepared pickle bundles must be trusted."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class MarketBundle:
    calendar: dict
    snapshots: dict
    units: dict
    histories: dict = field(default_factory=dict)
    covariances: dict | None = None
    returns: pd.DataFrame | None = None
    actions: dict = field(default_factory=dict)
    dividends: dict = field(default_factory=dict)

    def __post_init__(self):
        self.days = list(self.calendar["Days"])
        if self.days != sorted(set(self.days)):
            raise ValueError("Unsorted or duplicate calendar")
        self.dates = pd.DatetimeIndex(self.days)
        self.loc = {day: i for i, day in enumerate(self.days)}
        self.source_specs = {
            "baseline_vega": {"Book": "vega", "Env": {"INSTRUMENTS": "put"}},
            "baseline_variance": {"Book": "variance", "Env": {"INSTRUMENTS": "put"}},
        }
        for name, h in self.histories.items():
            required = {
                "SignalDate",
                "EntryDate",
                "ExitDate",
                "PlannedExit",
                "NetUSD",
                "NetForecast",
                "ForecastRisk",
                "Quantities",
            }
            if not required.issubset(h.columns):
                raise ValueError(f"Incomplete shadow history: {name}")
            if not ((h.SignalDate < h.EntryDate) & (h.EntryDate <= h.ExitDate)).all():
                raise ValueError("Invalid shadow dates")
            self.histories[name] = h.sort_values("SignalDate").reset_index(drop=True)

    @classmethod
    def load(cls, directory, data_root=None):
        directory = Path(directory)

        def read(name):
            path = directory / name
            if not path.exists():
                raise FileNotFoundError(f"Missing private input: {path}")
            return pd.read_pickle(path)

        histories = {}
        for book in ["vega", "variance"]:
            path = directory / (book + "_shadow_trades.pkl")
            if not path.exists():
                matches = list((directory / "sources" / ("baseline_" + book)).glob("*trades.pkl"))
                if len(matches) != 1:
                    raise FileNotFoundError(f"Missing {book} shadow history")
                path = matches[0]
            histories["baseline_" + book] = pd.read_pickle(path)
        covs = read("covariances.pkl") if (directory / "covariances.pkl").exists() else None
        if covs and "EWMA_0.94" in covs:
            covs = covs["EWMA_0.94"]
        actions = (
            json.loads((directory / "announced_actions.json").read_text())
            if (directory / "announced_actions.json").exists()
            else {}
        )
        dividends = (
            json.loads((directory / "dividends.json").read_text())
            if (directory / "dividends.json").exists()
            else {}
        )
        returns = None
        if data_root is not None:
            data_root = Path(data_root)
            returns = pd.read_csv(
                data_root / "processed/dow/dow_total_log_returns.csv", index_col=0, parse_dates=True
            )
            dividends = read_dividend_announcements(data_root)
        return cls(
            read("calendar.pkl"),
            read("snapshots.pkl"),
            {"post_quote": read("units.pkl"), "strictly_lagged": read("units_lagged.pkl")},
            histories,
            covs,
            returns,
            actions,
            dividends,
        )


def read_dividend_announcements(data_root):
    result = {}
    for path in (Path(data_root) / "raw/dow/dividends").glob("*.csv"):
        x = pd.read_csv(path)
        if "Dividend Ex Date" in x:
            result[path.stem] = [
                (str(r["Dividend Announcement Date"])[:10], str(r["Dividend Ex Date"])[:10])
                for _, r in x.iterrows()
                if pd.notna(r["Dividend Ex Date"])
            ]
    return result
