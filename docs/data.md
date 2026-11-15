# Private input specification

No raw or processed market inputs, credentials, quote caches, or pickles are included in this repository. The notebook and `reference` command require only the derived results in `reports/reference/`. A fresh market replay requires an independently supplied, licensed archive. Loading a prepared pickle bundle is appropriate only for trusted local files.

## Archive layout

Pass the directory containing `raw/` and `processed/` to `ewma-cal prepare --data-root`:

```text
processed/dow/dow_weights_daily.csv
processed/dow/dow_total_log_returns.csv
raw/dow/rates/dgs1mo.csv
raw/dow/prices/<RIC-with-dots-and-carets-replaced-by-underscores>.csv
raw/dow/dividends/<RIC>.csv
raw/dow/corporate_actions/<RIC>.csv
raw/dow/options/full_history/dataset_config.json
raw/dow/options/full_history/chains/<YYYY-MM-DD>/<symbol>.json
raw/dow/options/full_history/quotes/<OCC-option-symbol>.json
```

Weights require `Date,RIC,Weight`, with exactly 30 distinct members on every session. Total log returns have a date index and RIC columns, including historical members. Rates require `Date,RateDecimal`, with annual yields in decimal units. Price archives retain their original vendor columns; the current unit builder uses the total-return table to evolve the hedge underlying from the option quote's underlying price.

Option payloads are column-oriented JSON objects with `s: "ok"` and parallel arrays. The fields read by the model include `optionSymbol,side,expiration,strike,underlyingPrice,bid,ask,bidSize,askSize,volume,openInterest,updated`. Expiry accepts ISO dates or the original vendor timestamp. `updated` is a Unix timestamp, interpreted in New York time. Quote histories contain all observed records for the contract; the archive reader keeps the latest valid quote per session.

Corporate-action CSV fields include `Instrument`, `Corporate Change Event Type`, `Capital Change Is Rescinded`, `Capital Change Announcement Date`, `Capital Change Ex Date`, and `Capital Change Effective Date`. Dividend files use `Dividend Announcement Date`, `Dividend Ex Date`, and `Gross Dividend Amount`. Exclusions use events announced by the signal date, not subsequently discovered events. Preserve historical symbols and delisted constituents; do not replace them with today's membership.

`prepare` builds snapshots, EWMA matrices, both hedge paths, and both shadow histories. It writes a private bundle with:

- `calendar.pkl`: `Days`, `Rates`, and point-in-time `Members`.
- `snapshots.pkl`: per-date option/underlying state including exact quote pairs.
- `covariances.pkl`: per-date stock names, EWMA matrix, and observation counts.
- `units.pkl` and `units_lagged.pkl`: per-contract, per-side mark-to-market paths, spread costs, hedge turnover, and funding exposures.
- `vega_shadow_trades.pkl` and `variance_shadow_trades.pkl`: exact-position opportunities, forecasts, quantities, and eventual outcomes.
- `announced_actions.json` and `dividends.json`: as-announced event calendars.

The replay loader also accepts the historical research cache layout with `sources/baseline_vega/*trades.pkl` and `sources/baseline_variance/*trades.pkl`. It never searches outside the supplied input directory.

## Benchmark

Supply a CSV with `Date,AdjustedClose`, one unique, positive adjusted close per strategy valuation date. Obtain SPY total-return adjusted history from a provider licensed for your use. The saved reference benchmark source and checksum are recorded alongside the model manifest. Do not substitute unadjusted closes while retaining the total-return label.

Keep inputs and generated replay results under ignored `data/` and `outputs/` directories. Only derived, reviewed reporting artifacts belong in `reports/reference/`.

## Redistribution

Processing or changing a file format does not itself create redistribution rights. Keep source quotes, prices, constituent histories, feature matrices, calibrated market-state caches and prepared bundles private unless the applicable licence explicitly permits publication. Published portfolio/P&L summaries are derived research outputs; their presence does not grant rights to the underlying data. Source: [Market Data](https://www.marketdata.app) for historical options, and LSEG for the private constituent and underlying-data inputs. The SPY benchmark input comes from Yahoo Finance via yfinance.
