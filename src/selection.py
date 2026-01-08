# src/selection.py
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import List, Tuple

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller


@dataclass(frozen=True)
class SelectionConfig:
    """
    Pair selection configuration.

    Learning notes:
    - corr_threshold is a *cheap filter* to avoid testing too many nonsense pairs.
    - adf_pvalue_threshold is used to screen for (approx) stationarity of residual spread.
    - half-life range filters out spreads that revert too slowly (not tradable) or too fast/noisy.
    """
    corr_threshold: float = 0.70
    adf_pvalue_threshold: float = 0.05
    half_life_min: float = 2.0
    half_life_max: float = 30.0
    zscore_lookback: int = 60  # used later in signals; included here for context
    top_n: int = 5


def all_pairs(tickers: List[str]) -> List[Tuple[str, str]]:
    """
    Generate all unordered pairs of tickers.
    """
    tickers = [t.strip().upper() for t in tickers]
    return list(combinations(tickers, 2))


def _log_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Log-transform prices (safer for ratio/spread modelling).
    """
    if (prices <= 0).any().any():
        raise ValueError("Prices must be positive to compute log prices.")
    return np.log(prices.astype(float))


def _log_returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns from prices.
    """
    lp = _log_prices(prices)
    return lp.diff().dropna()


def prefilter_by_corr(
    returns: pd.DataFrame,
    threshold: float = 0.70,
) -> pd.DataFrame:
    """
    Compute correlation matrix and return a long dataframe of pairs above threshold.

    Returns a DataFrame with columns:
    - a, b, corr
    """
    corr = returns.corr()
    tickers = list(corr.columns)

    rows = []
    for i, a in enumerate(tickers):
        for b in tickers[i + 1 :]:
            c = corr.loc[a, b]
            if np.isfinite(c) and c >= threshold:
                rows.append((a, b, float(c)))

    out = pd.DataFrame(rows, columns=["a", "b", "corr"])
    out = out.sort_values("corr", ascending=False).reset_index(drop=True)
    return out


def fit_hedge_ratio(logA: pd.Series, logB: pd.Series) -> Tuple[float, float]:
    """
    Fit logA = alpha + beta * logB + eps using OLS.

    Returns:
        alpha, beta
    """
    df = pd.concat([logA.rename("A"), logB.rename("B")], axis=1).dropna()
    y = df["A"].values
    X = add_constant(df["B"].values)  # [1, logB]
    model = OLS(y, X).fit()
    alpha = float(model.params[0])
    beta = float(model.params[1])
    return alpha, beta


def compute_residual(logA: pd.Series, logB: pd.Series, alpha: float, beta: float) -> pd.Series:
    """
    Residual/spread: eps_t = logA_t - (alpha + beta * logB_t)
    """
    df = pd.concat([logA.rename("A"), logB.rename("B")], axis=1).dropna()
    eps = df["A"] - (alpha + beta * df["B"])
    eps.name = "residual"
    return eps


def adf_pvalue(residual: pd.Series) -> float:
    """
    Augmented Dickey-Fuller test p-value on residual.
    Null: residual has a unit root (non-stationary).
    """
    x = residual.dropna().values
    if len(x) < 50:
        return float("nan")
    try:
        # autolag chooses lag based on information criterion
        res = adfuller(x, autolag="AIC")
        return float(res[1])
    except Exception:
        return float("nan")


def estimate_half_life(residual: pd.Series) -> float:
    """
    Estimate half-life of mean reversion using a simple AR(1)-style approximation.

    Standard approach:
      Δx_t = phi * x_{t-1} + noise
    If phi < 0 => mean-reverting, half-life = -ln(2) / phi

    Returns:
        half-life in days, or nan if not mean-reverting / insufficient data
    """
    x = residual.dropna()
    if len(x) < 50:
        return float("nan")

    x_lag = x.shift(1).dropna()
    dx = (x - x.shift(1)).dropna()

    # Align indices
    dx = dx.loc[x_lag.index]

    # Regress dx on x_{t-1}
    X = add_constant(x_lag.values)
    y = dx.values
    try:
        model = OLS(y, X).fit()
        phi = float(model.params[1])
    except Exception:
        return float("nan")

    # Mean reversion requires phi < 0 in this formulation
    if not np.isfinite(phi) or phi >= 0:
        return float("nan")

    half_life = -np.log(2) / phi
    if half_life <= 0 or not np.isfinite(half_life):
        return float("nan")
    return float(half_life)


def evaluate_pair(
    prices_train: pd.DataFrame,
    a: str,
    b: str,
) -> dict:
    """
    Compute diagnostics for a single pair inside a training window.

    Returns dict with:
      a, b, corr, alpha, beta, adf_p, half_life, resid_std
    """
    # Returns correlation filter uses log returns
    rets = _log_returns_from_prices(prices_train[[a, b]])
    corr = float(rets[a].corr(rets[b]))

    # Hedge ratio + residual are fitted on log prices
    lp = _log_prices(prices_train[[a, b]])
    alpha, beta = fit_hedge_ratio(lp[a], lp[b])
    resid = compute_residual(lp[a], lp[b], alpha, beta)

    return {
        "a": a,
        "b": b,
        "corr": corr,
        "alpha": alpha,
        "beta": beta,
        "adf_p": adf_pvalue(resid),
        "half_life": estimate_half_life(resid),
        "resid_std": float(resid.std(ddof=1)),
        "n_obs": int(len(resid)),
    }


def select_pairs(
    prices_train: pd.DataFrame,
    cfg: SelectionConfig = SelectionConfig(),
) -> pd.DataFrame:
    """
    Select and rank pairs for a given training window.

    Process:
    1) Compute log returns corr and prefilter pairs.
    2) For each candidate, fit hedge ratio, compute residual, run ADF, estimate half-life.
    3) Filter by ADF p-value and half-life range.
    4) Rank: lowest ADF p-value, then corr (desc), then half-life (closest to mid band).

    Returns:
      DataFrame with diagnostics, sorted best-first.
    """
    # Safety: require at least 2 columns
    if prices_train.shape[1] < 2:
        raise ValueError("Need at least 2 tickers in prices_train.")

    # Step 1: prefilter pairs by correlation
    returns = _log_returns_from_prices(prices_train)
    corr_pairs = prefilter_by_corr(returns, threshold=cfg.corr_threshold)

    if corr_pairs.empty:
        return pd.DataFrame(columns=["a", "b", "corr", "alpha", "beta", "adf_p", "half_life", "resid_std", "n_obs"])

    # Step 2: evaluate candidates
    rows = []
    for _, row in corr_pairs.iterrows():
        a, b = row["a"], row["b"]
        try:
            stats = evaluate_pair(prices_train, a, b)
            rows.append(stats)
        except Exception:
            # If something fails (data alignment, regression issues), skip the pair.
            continue

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Step 3: filter by stationarity and half-life
    out = out.dropna(subset=["adf_p", "half_life"])
    out = out[out["adf_p"] <= cfg.adf_pvalue_threshold]
    out = out[(out["half_life"] >= cfg.half_life_min) & (out["half_life"] <= cfg.half_life_max)]

    if out.empty:
        return out

    # Step 4: rank
    target_hl = 0.5 * (cfg.half_life_min + cfg.half_life_max)
    out["hl_distance"] = (out["half_life"] - target_hl).abs()

    out = out.sort_values(
        by=["adf_p", "corr", "hl_distance"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    return out.head(cfg.top_n)
