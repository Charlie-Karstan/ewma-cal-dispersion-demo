"""As-announced contract-adjustment event calendar, including legacy issuer RICs."""

import re
from pathlib import Path

import pandas as pd


def announced_actions(root):
    directory = Path(root) / "raw/dow/corporate_actions"
    if not directory.exists():
        raise FileNotFoundError(f"Missing corporate-action archive: {directory}")
    actions = {}
    for path in (Path(root) / "raw/dow/corporate_actions").glob("*.csv"):
        x = pd.read_csv(path, dtype=str)
        for _, row in x.iterrows():
            kind = str(row.get("Corporate Change Event Type", "")).lower()
            if not any(k in kind for k in ["split", "consolidation", "demerger", "exchange"]):
                continue
            if str(row.get("Capital Change Is Rescinded", "")).lower() == "true":
                continue
            ann = str(row.get("Capital Change Announcement Date", ""))[:10]
            ric = str(row.get("Instrument", ""))
            for col in ["Capital Change Ex Date", "Capital Change Effective Date"]:
                event = str(row.get(col, ""))[:10]
                if (
                    re.fullmatch(r"\d{4}-\d{2}-\d{2}", ann)
                    and re.fullmatch(r"\d{4}-\d{2}-\d{2}", event)
                    and ann < event
                ):
                    actions.setdefault(ric, set()).add((ann, event))
    return {ric: sorted(events) for ric, events in actions.items()}
