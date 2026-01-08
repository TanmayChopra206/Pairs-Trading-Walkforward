Statistical Pairs Trading with Walk-Forward Validation (US Banks)
Overview

This project implements a full end-to-end statistical pairs trading framework on US bank equities, with a strong emphasis on out-of-sample robustness and realistic assumptions.

Rather than optimising for in-sample profitability, the goal is to understand when and why statistical arbitrage works — and when it fails.

Methodology
1. Data

Daily adjusted close prices for large US banks (e.g. BAC, JPM, PNC, USB, TFC)

Source: Yahoo Finance (yfinance)

Log-price modelling throughout

2. Pair Selection (Training Window)

Pairs are selected dynamically using:

Minimum correlation threshold

Cointegration testing (ADF on residuals)

Mean-reversion half-life constraints

This ensures selected pairs are statistically justified rather than visually appealing.

3. Signal Generation

For each selected pair:

Hedge ratio estimated via linear regression on log prices

Spread constructed as log-price residual

Rolling z-score computed

Entry/exit signals generated using symmetric thresholds

Positions lagged by one day to avoid look-ahead bias

4. Backtesting

Spread-based portfolio returns

Explicit transaction costs (bps per unit turnover)

No re-optimisation during test periods

Performance evaluated using PnL, Sharpe ratio, drawdowns, and hit rate

5. Portfolio Construction

Top-N selected pairs traded simultaneously

Equal-weight aggregation of pair returns

Diversification effects explicitly analysed

6. Walk-Forward Validation

Expanding training window (3 years)

Rolling test window (1 year)

Parameters re-estimated at each step

Portfolio returns stitched across folds to produce a single out-of-sample equity curve

This eliminates look-ahead bias and highlights regime dependence.

Key Findings

Mean-reversion signals are not stable across regimes

Strategy underperforms during calm, trending markets

Performance improves significantly during market dislocations (e.g. 2020 crisis)

Portfolio diversification reduces drawdowns but does not eliminate regime risk

Transaction costs materially impact short-horizon statistical strategies

Overall, the strategy behaves more like a relative-value crisis alpha than a steady carry trade.

Conclusion

This project demonstrates that statistically sound signals do not guarantee persistent profitability. Robust research requires:

Walk-forward validation

Honest treatment of costs

Explicit analysis of failure modes

The framework prioritises process correctness over headline Sharpe ratios.

Technologies

Python (NumPy, pandas, statsmodels)

yfinance

matplotlib

Modular research architecture
