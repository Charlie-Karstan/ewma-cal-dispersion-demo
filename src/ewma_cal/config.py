"""Frozen model settings; execution stresses never alter signal parameters."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    name: str = "prof2_EWMA_CAL"
    version: str = "2.1.0"
    capital: float = 2_000_000.0
    vega_gate: float = 0.15
    hedge_timing: str = "strictly_lagged"

    def __post_init__(self):
        if self.capital <= 0:
            raise ValueError("Capital must be positive")
        if self.hedge_timing not in {"strictly_lagged", "post_quote"}:
            raise ValueError("Unknown hedge timing")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))
