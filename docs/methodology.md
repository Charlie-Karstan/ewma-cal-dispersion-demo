# Model and accounting

## Forecast and position construction

1. Use the point-in-time Dow membership and weights, with DIA as the index leg. Choose a common expiry near 30 calendar days (allowed range 14–70 days), and near-ATM call/put pairs with valid quoted markets. A complete 30-stock snapshot is required to assess baskets; the traded basket can be smaller.
2. Estimate daily stock covariance using an EWMA decay of 0.94 and at most 252 trailing sessions. Missing-pair normalization, an implied-volatility prior for short histories, correlation shrinkage, and a positive-semidefinite repair stabilize the matrix. Index covariance is projected from constituent weights.
3. Forecast the next-session entry and the exit five sessions after entry. Integrate the covariance over that horizon and value each exact strike with remaining time value. This is not a one-day correlation threshold applied to a five-day trade. The vega sleeve mean-reverts implied variance toward forecast variance plus a lagged variance-premium estimate at rate 0.3; the variance sleeve holds implied volatility constant (rate 0). The premium estimate updates with decay 0.97 using the signal-date gap. Pricing uses a European approximation for the quoted American contracts.
4. Evaluate both dispersion directions, 5/10/15/30-stock baskets, and weight, execution-cost, and expected-payoff rankings. Choose quantities and basket together. The index instrument is a **put**, while constituent instruments are straddles. This is an asymmetric dispersion overlay, not a perfectly vega-and-gamma-neutral index-straddle portfolio. The integer sleeve targets vega matching; the variance sleeve uses variance-replication quantities, including fractional equivalents. The relative tracking-error ceiling is 0.75, maximum basket concentration is 0.40, and variance coverage must reach 0.85 in the variance sleeve. The integer sleeve allows 20% vega mismatch and 100% beta-adjusted gamma mismatch. Each source builder retains its $1 million sizing budget and gross-notional ceiling of twice that budget; funding and position sizing are unchanged. The two sleeves are independently financed against the fixed $2 million portfolio denominator; cash interest never increases position sizes.
5. Score exact-position expected P&L after the modeled costs, normalized by forecast risk. Each sleeve maintains a non-overlapping shadow opportunity stream. A completed shadow opportunity contributes a calibration label even when the calibrated strategy did not trade it. Opportunity availability is therefore determined by the shadow stream, not solely by previously accepted live entries. As-announced corporate actions and dividend dates screen incompatible holdings. The historical development/later-period boundary excludes shadow positions crossing into 2022; this research convention is retained rather than refitted.

## EWMA-CAL versus standard EWMA

The physical covariance forecast remains EWMA. The additional layer learns **realized exact-position net payoffs**, not just returns or correlations. For signal date `t`, only shadow outcomes with `ExitDate < t` may train the model. Use the latest 40 completed outcomes, with at least five labels, and normalize each realized and predicted payoff by its forecast risk.

- **Vega sleeve:** EWMA decay 0.80 over normalized realized payoffs. Enter when the weighted mean exceeds **0.15**. The dollar payoff forecast equals this mean times current forecast risk.
- **Variance sleeve:** EWMA decay 0.94 over realized payoffs and forecast residuals. Blend 50% of the historical normalized payoff estimate with 50% of the current option forecast plus the estimated residual correction. Enter when the resulting dollar forecast is positive.

The empirical calibration is the source of the model's claimed research edge: it conditions entry on previously observed profitability of comparable opportunities and corrects option-forecast bias. Positive development performance is evidence within the research sample, not proof of independent alpha. Gate 0.15 was chosen during research; the later period has subsequently been inspected.

The labels remain the frozen **post-quote shadow payoff target**, including when the replay uses lagged hedge execution. Strictly lagged execution changes realized fills and hedge P&L, without refitting the signal. This is intentional and disclosed rather than silently changing the trained target.

## Execution and costs

The release executes a hedge using the previously observable hedge target. The comparison convention updates the target using the current quote. Signals arise before next-session option entry. Missing planned exits defer until all legs have observable marks; entered trades are never removed because the eventual exit is inconvenient. Missing intermediate marks remain missing and suppress daily risk statistics rather than being filled with zero returns.

| Item | Frozen convention |
|---|---|
| Option spread | Reference: midpoint; stresses charge 0.25, 0.50, or 1.00 times entry-plus-exit quoted half-spreads |
| Option commission | $0.65 per contract per side; variance equivalents retain minimum-ticket charges |
| Stock execution | 1 basis point of hedge dollar turnover plus $0.005 per share |
| Short-stock borrow | 1% annualized on short-stock exposure |
| Long-stock funding | Treasury yield plus 3% annualized |
| Long-option debit funding | Treasury yield, already included in the base replay |
| Variance handling | $0.35 per nonzero leg per modeled hedge session, including delayed exits |
| Market depth | Quotes must pass initial quality filters; strict quantity-at-displayed-size enforcement is not the reference assumption |

A spread fraction of zero does **not** mean zero total trading costs. The full quoted-spread stress is a separate execution scenario, not an assertion that midpoint execution can always be obtained. Position size and calibration labels remain fixed throughout these stresses. The existing source depth conventions and borrow assumptions are not a broker margin or stock-locate model.

## Portfolio and forecasting metrics

All reports use $2 million initial capital and common daily dates. Full history spans January 2016–November 2026, with source opportunities beginning in 2018 after the snapshot/label warm-up. Development is 2018–2021; “OOS” denotes 2022–2026, whose results have already been examined. Each subperiod resets the capital and interest ledger at its first valuation; its return excludes prior gains. The full-period path does not reset. Portfolio metrics mark positions daily; payoff forecasting metrics use entered positions assigned by entry date, not daily NAV observations.

- **CAGR:** calendar-time annualized endpoint return. Total return and dollar P&L are also shown.
- **Sharpe:** mean daily simple return divided by sample volatility, annualized by sqrt(252), with zero benchmark. Treasury-excess Sharpe subtracts the prior-rate accrual over the actual calendar interval. Cash-rich raw Sharpe can therefore be high without strong trading alpha.
- **Drawdown / Calmar:** minimum peak-to-trough NAV return; CAGR divided by the absolute maximum drawdown. Sortino uses the root mean square of negative daily returns with zero threshold.
- **Information ratio:** daily portfolio return minus the SPY total-return proxy, annualized by sqrt(252). Beta and correlation use the same aligned returns. This is a directional-equity comparison, not a mandate-matched dispersion benchmark. Treasury collateral is also shown.
- **Payoff errors:** MSE in dollars squared; RMSE and MAE in dollars; **ASE means average squared error normalized by forecast risk**, `mean(((prediction-realization)/risk)^2)`. Bias is mean prediction minus realization. Pearson and Spearman IC compare predicted with realized net trade payoffs. These are conditional entered-trade errors, not physical covariance-forecast MSE or an all-opportunity forecast score.
- Volatility, tail loss, trade count, hedge turnover, margin proxy, net vega/gamma extrema, annual returns, and cash attribution are in the saved outputs. The margin estimate is a conservative research proxy, not a broker-approved capital requirement.

Cash interest uses ACT/365.25 and the prior observed one-month Treasury rate on settled collateral. Only trade settlements change the interest-bearing balance; unrealized gains do not. The trading funding charges remain in trading P&L. This assumes a separately financed overlay whose collateral is remunerated; it is not a simulation of a specific brokerage cash account. The notebook distinguishes trading returns, trading plus cash, and Treasury-only returns.

SPY adjusted closes incorporate distributions and ETF expenses and proxy the S&P 500 total return. Missing benchmark dates are rejected rather than forward-filled. Aggregate normalized performance paths are saved; the original adjusted-close input stays private.

## Frozen funded sizing overlay (2.2.0)

The core model above is retained. The new identity is `prof2_EWMA_CAL_size8_cap4m_funded`. Target scale is 8, initial capital remains $2 million, and the aggregate margin-proxy cap is $4 million. This resizing supersedes the source builders' original per-sleeve sizing budgets; those budgets describe the unscaled core, not the new aggregate risk limit.

At each entry session, compute remaining proxy capacity after the positions still active on that session. Positions entering together share a common scale `min(8, available_capacity / sum(base_margin))`. The cap can reduce quantities below the target or block an entry. All legs and delta hedges scale together. Option-equivalent quantities can be fractional. Option commissions retain the existing ticket minimums, and handling charges remain per leg/session; other leg costs scale with exposure. No forecast or calibration parameter changes.

For each calendar interval, extra funding is `max(previous_margin_proxy - initial_capital, 0) * (previous_Treasury_rate + 0.03) * elapsed_days / 365.25`. It is allocated to the positions responsible for that prior exposure. The final modeled interval settles with the position's existing terminal cashflow, matching the frozen sizing-research convention, including the interval immediately following its final recorded position session. This explicit convention is retained rather than silently changing the saved result. The extra charge is a conservative funding stress on top of the leg funding already present. It is not a brokerage credit agreement or a dynamic NAV margin requirement.

The net trade contribution, cumulative portfolio P&L, settled cash ledger, and additional funding allocations reconcile. Every funding interval uses a previously observed rate and exposure. The calibration forecast still targets the scaled shadow payoff **before** this additional collateral-funding stress; entered-trade forecasting errors compare that prediction against the funded realization. Dollar MSE changes mechanically with sizing; risk-normalized ASE is also supplied.

The leverage sensitivity fixes every model and cost parameter and the $4m cap, varying only the target multiple. Development, full and known later-window results are all reported, with and without cash. No sensitivity result changes the frozen 8x default. Because larger-size execution and the later window have already been examined, this is a retrospective research comparison. Displayed-depth statistics describe whether all legs fit quoted entry size; they do not establish the ability to obtain larger midpoint fills over time.
