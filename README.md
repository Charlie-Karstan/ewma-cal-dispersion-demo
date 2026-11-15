# EWMA-CAL Dispersion Demo

A research demo with an executed notebook, saved aggregate results, and reproducible strategy code. Market-data inputs are excluded. The notebook runs from the included reporting artifacts; a fresh market replay requires separately supplied licensed inputs.

**Frozen model: `prof2_EWMA_CAL_size8_cap4m_funded`, version 2.2.0.** The original gate-0.15 signals and strictly lagged hedging are retained. Target quantities are multiplied by **8**, then clipped at entry to a **$4 million aggregate margin proxy** against **$2 million initial capital**. Extra collateral above initial capital is charged the prior observed Treasury yield **plus 3% annually**, in addition to the existing leg-level costs.

The 8x target was fixed before the accompanying leverage comparison. It is a multiple of the original position quantities, **not a promise of 8x portfolio leverage**. The cap frequently reduces actual size or prevents a new entry. Fractional option equivalents and hypothetical midpoint execution remain research assumptions. This margin proxy is not a broker-approved margin model.

## Run the analysis

Python 3.12 or newer (reference generated with Python 3.13):

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,notebook]'
ewma-cal reference
pytest -q
```

For exact direct dependency versions, install `requirements-lock.txt` before the editable package. Open [the executed notebook](notebooks/prof2_EWMA_CAL.ipynb) for saved portfolio metrics, S&P 500 comparisons, cash attribution, payoff errors, execution stress, and the leverage grid. It runs from committed **aggregate backtest results** without market inputs or network access.

## Reproduce from private inputs

Licensed raw/processed inputs and caches are excluded. The [input specification](docs/data.md) describes the required archive.

```sh
ewma-cal prepare --data-root /path/to/private/data --output outputs/private_inputs
pip install -e '.[benchmark]'
ewma-cal benchmark --output outputs/benchmark.csv

ewma-cal sized-report --inputs outputs/private_inputs --benchmark outputs/benchmark.csv \
  --config config/model.json --sizing config/sizing.json --output outputs/funded_replay

ewma-cal leverage-test --inputs outputs/private_inputs --benchmark outputs/benchmark.csv \
  --config config/model.json --sizing config/sizing.json --output outputs/leverage
```

The leverage test varies only target multiples **0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16**. It holds the $4m cap, financing spread, signals, hedge convention and midpoint option-fill assumption fixed. It reports full, development, OOS periods, with and without cash interest. **It does not select a replacement default.** The frozen file checksums are verified before and after the comparison.

`ewma-cal report` remains the original unscaled-core replay. Use `sized-report` to reproduce this funded version.

## Interpretation

Cash collateral earns the previous observed one-month Treasury yield on settled balances. Interest is reinvested without resizing the trading overlay and is reported separately. Raw Sharpe, Treasury-excess Sharpe, Calmar, benchmark information ratio, drawdown, trade concentration, and forecasting errors are available in the notebook and CSVs.

SPY adjusted close is a **total-return proxy for the S&P 500**, with common dates and initial capital. The later period has already been examined; neither it nor the leverage comparison is independent validation. Larger midpoint fills have not been established by this archive. Entry-depth diagnostics and 25%, 50%, and 100% quoted-half-spread stresses remain visible.

## Layout and provenance

- `src/ewma_cal/`: local market preparation, frozen forecasting/calibration, position replay, cash accounting, funded sizing, and reporting.
- `config/model.json`: unchanged core settings; `sizing.json`: frozen sizing; `freeze.json`: model identity and file checksums.
- `reports/reference/`: funded 8x-model results, benchmark provenance, and verification.
- `reports/leverage/`: fixed-grid comparison; target 8 remains the frozen default.
- `notebooks/`: executed analysis; `tests/`: accounting, timing, capacity, and funding checks.

See [methodology](docs/methodology.md) for the equations, cost assumptions and funding settlement convention. The earlier 2.1.0 release and all local research branches remain preserved.

## Data sources and publication

Source: [Market Data](https://www.marketdata.app) for the privately held historical option archive; LSEG for private constituent, return and corporate-action inputs; Yahoo Finance via yfinance for the SPY benchmark input. No provider endorsement is implied.

The public files contain code and derived research reporting artifacts, not the underlying raw or processed market-input datasets. A transformed price/quote table is not automatically licensed for redistribution. Third-party data rights remain governed by the applicable provider and exchange agreements. See [the publication audit](docs/publication.md) for the reviewed scope and reproducibility limits.

LaTeX-ready chart exports (vector PDF and 300-dpi PNG) are in [reports/figures](reports/figures/README.md), covering implied/realized correlation, sleeve-level payoff predictions, and equity/drawdown.
