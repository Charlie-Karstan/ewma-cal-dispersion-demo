# Publication and reproducibility audit

The public repository was inspected from an independent clone, including all published branches, tags and their reachable file history. The private research repository and its market-data history are not ancestors of the published release.

## Published material

The repository contains strategy source, configurations, tests, documentation, executed notebook outputs and derived backtest reports. CSVs contain portfolio equity, normalized benchmark performance, trade-level P&L/forecast labels, portfolio metrics sensitivity summaries and monthly aggregated correlation diagnostics. They are **derived reporting artifacts**, not the raw or processed market-input datasets used by the engine.

No raw API responses, per-contract quote histories, bid/ask/underlying-price tables, constituent/return panels, option-state matrices, prepared input bundles, pickles, API credential files or private keys were found. Pattern scans of every reachable text blob and notebook text found no credential matches. Embedded images were separated from notebook text for this scan; file types and every historical CSV header were also reviewed. This records the scan's scope and findings rather than a guarantee that a pattern scanner detects every possible secret.

## Functional verification

A new isolated Python 3.13 environment installed the package using the documented extras. All 16 tests, formatting checks, reference-report commands and frozen-source/configuration checks passed. The notebook was re-executed using the public clone alone, without the private market archive. This exposed a missing Jinja2 dependency for formatted pandas tables; packaging patch 2.2.1 declares it and CI now executes the notebook as well as the tests. The numerical model, freeze hashes and saved results are unchanged.

The market-data preparation and full replay require independently supplied, licensed inputs. They cannot be reproduced solely from the public reporting artifacts. Earlier verification against the private archive is documented in `reports/reference/verification.json`. Historical tags preserve earlier packaging; the current branch includes the notebook dependency fix.

## Data rights

Source: [Market Data](https://www.marketdata.app) for historical options; LSEG for privately held constituent/underlying inputs; Yahoo Finance via yfinance for the SPY benchmark input. No provider endorsement is implied.

Processing data or converting it into CSV does not itself grant redistribution rights. Market Data's [educational historical-data terms](https://www.marketdata.app/terms/public-use/) distinguish aggregated findings from confidential underlying datasets, require attribution, and condition publication on the applicable scope. Its [general terms](https://www.marketdata.app/terms/) also refer derived-data rights to exchange agreements. Other providers and exchanges have their own terms. This repository does not establish the owner's account-specific redistribution permissions.

Only derived research reports are published here. The source archive and processed inputs remain private. The inclusion of reporting artifacts is not a licence to reuse or redistribute any underlying vendor data.

## Research interpretation

This is a research demo with a frozen model and retrospective sensitivities. It uses hypothetical midpoint fills, fractional option equivalents and a margin proxy. Larger fills remain unverified. The later window has been examined; it is not independent validation. Cash income is separately identified, and the published forecasting metrics are entered-payoff errors, not covariance forecast accuracy. These limitations remain visible in the README, methodology and notebook.
