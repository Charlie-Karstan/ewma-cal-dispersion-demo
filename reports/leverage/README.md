# Fixed-cap leverage comparison

Frozen default: 8x target, $4m aggregate margin proxy, $2m initial capital, extra funding at the prior Treasury yield plus 3%. Signals, calibration, hedge timing, and midpoint-fill assumptions are held fixed. No new default was selected.

| Target size | Full trading CAGR | Full CAGR with cash | Trading Sharpe | Trading max DD | Known later trading CAGR | Trades |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5x | 0.16% | 2.49% | 0.578 | -0.49% | -0.02% | 116 |
| 1x | 0.35% | 2.68% | 0.650 | -0.93% | -0.01% | 116 |
| 2x | 0.71% | 3.04% | 0.694 | -1.76% | 0.02% | 116 |
| 3x | 1.06% | 3.38% | 0.712 | -2.51% | 0.04% | 116 |
| 4x | 1.36% | 3.68% | 0.703 | -3.28% | -0.01% | 116 |
| 6x | 1.72% | 4.04% | 0.625 | -4.72% | -0.38% | 116 |
| 8x | 2.05% | 4.36% | 0.625 | -4.96% | -0.43% | 115 |
| 10x | 2.11% | 4.42% | 0.607 | -5.02% | -0.39% | 114 |
| 12x | 2.08% | 4.39% | 0.572 | -5.10% | -0.41% | 114 |
| 16x | 1.98% | 4.29% | 0.514 | -5.11% | -0.41% | 114 |

Three times target size has the strongest full-period trading Sharpe in this grid; 10x has the highest full-period trading CAGR. Neither observation constitutes independent model selection. The frozen 8x version is retained pending a final choice. The results include the higher risk, funding expense and execution assumptions; the larger exposures do not improve the forecast engine.

The target is a multiplier of original option-equivalent quantities, not a fixed portfolio leverage ratio. Mean realized scale and clipped-entry fractions are reported in overview.csv. Different targets can block or admit different entries as current capacity becomes available.

Both cash treatments and all three periods appear in leverage_metrics.csv. It includes zero-benchmark and Treasury-excess Sharpe, information ratio versus the SPY total-return proxy, Calmar, drawdown, funding charges, cash attribution, concentration and displayed-depth diagnostics. leverage_forecasts.csv contains entered-payoff MSE, RMSE, MAE and risk-normalized ASE by sleeve and jointly. It evaluates the scaled shadow prediction against funded realized payoffs.

The default 8x row reproduces the frozen funded model across every period and both cash treatments. The 1x row matches the previous core release. Frozen configuration hashes are checked before and after the test. Larger midpoint fills remain unverified, and fractionally scaled option equivalents remain a research approximation.
