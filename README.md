# Statistical Pairs Trading with Walk-Forward Validation (US Banks)

A **statistical arbitrage framework** designed to study **mean-reversion robustness, regime dependence, and failure modes** in US bank equities using strict out-of-sample validation.

---

## Project Overview

In This project I implement a full pairs trading pipeline with an emphasis on **realistic deployment constraints** rather than in-sample performance optimisation.  
The objective is not to maximise Sharpe in hindsight, but to understand **when and why relative-value signals work and when they break**.

---

##  Methodology

### 1. Data & Universe
- **Universe:** Large-cap US banks (e.g. `JPM`, `BAC`, `PNC`, `USB`, `TFC`)
- **Frequency:** Daily adjusted close prices
- **Source:** `yfinance`
- **Transformation:** Log-prices used throughout to stabilise relative dynamics

---

### 2. Pair Selection
Pairs are selected during each training window using:

- **Correlation filtering** to ensure shared market drivers
- **Cointegration testing (ADF)** on regression residuals
- **Mean-reversion speed constraints** via half-life estimation
- **Ranking** by statistical quality and stability metrics

All selection criteria are **re-estimated at each walk-forward step** to avoid parameter leakage.

---

### 3. Signal Construction
- Hedge ratios estimated via linear regression
- Spread normalised using rolling z-scores
- Entry/exit rules defined symmetrically around zero
- Signals are **lagged by one day** to prevent look-ahead bias

---

### 4. Backtesting & Execution Assumptions
- Equal-weight allocation across selected pairs
- Dollar-neutral construction at the pair level
- Transaction costs applied in basis points per unit turnover
- P&L aggregated both **per-pair** and **portfolio-level**

---

### 5. Walk-Forward Validation
A rolling validation framework is used to assess robustness:

- **Training window:** 3 years (expanding)
- **Test window:** 1 year (rolling)
- **Re-selection:** Pairs and parameters re-estimated at each fold

This structure explicitly tests performance under **structural breaks and regime shifts**.

---

## Key Findings

- **Strong regime dependence:** Mean-reversion is unreliable in trending, low-volatility environments
- **Crisis sensitivity:** Relative-value signals perform best during dislocations (e.g. 2020)
- **Transaction costs dominate:** Many statistically “clean” signals fail after realistic frictions
- **Low hit-rate strategies can still work**, but only when drawdowns are controlled

Overall, the strategy behaves more like a **relative-value hedge** than a consistent alpha source.

---

##  Limitations & Next Steps

- Pair stability degrades significantly outside stress regimes
- Static thresholds struggle in changing volatility environments
- Next extensions:
  - Volatility-scaled position sizing
  - Regime classification filters
  - Cross-sector relative-value expansion

---

## Technologies
- **Python:** pandas, NumPy, statsmodels
- **Data:** yfinance
- **Visualisation:** matplotlib, seaborn
