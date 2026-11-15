# LaTeX figures — frozen funded DIA-put model

Use the PDF exports with `graphicx`: they preserve vector lines and embedded fonts. Matching PNGs are provided at 300 dpi.

- `implied_realized_correlation`: monthly averages of the implied correlation proxy and five-session realized correlation, with an implied-minus-realized panel. The average daily paired gap is 4.40 percentage points across 2,047 observations. Implied IVs reflect the selected option tenor (generally near 30 days); realized correlation covers t+2 through t+6. This is a descriptive comparison with different tenors, not a matched-tenor risk-premium estimate. Full point-in-time 30-stock weights are used, rather than the smaller traded basket. The proxy uses constituent variances and index variance to infer a common pairwise correlation. Realized covariance uses uncentered return second moments, consistent with the existing forecast diagnostics. The displayed gap mean weights daily observations equally; plotted curves are monthly means. Overlapping outcomes are not independent observations.
- `predicted_realized_payoff`: entered trades, split into vega and variance sleeves. Common limits across sleeves, dollar values in thousands, and a perfect-prediction line. Predictions precede the additional collateral-funding stress; realized net payoffs include it. No points are trimmed. Colors indicate Development and OOS.
- `equity_drawdown`: continuous full-period portfolios with and without collateral interest, and SPY adjusted total return as the S&P 500 proxy. Initial values are normalized to 100. The equity axis is logarithmic; drawdowns use the daily running maximum.

Source: [Market Data](https://www.marketdata.app) options, LSEG underlying returns/weights, and Yahoo Finance via yfinance for the SPY benchmark. Only aggregated correlation diagnostics and derived performance outputs are included; source market-input datasets remain private. Figures are descriptive research outputs, with the model's existing execution and validation qualifications unchanged.

The notebook regenerates all six exports. From the repository root, they can also be regenerated with:

```python
from ewma_cal.figures import export_figures
export_figures('.')
```

Example LaTeX (copy the PDFs alongside the document):

```latex
\usepackage{graphicx}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=\linewidth]{equity_drawdown.pdf}
  \caption{Frozen funded DIA-put model: equity and drawdown,
    with and without collateral interest and versus the SPY total-return proxy.}
  \label{fig:equity-drawdown}
\end{figure}
```
