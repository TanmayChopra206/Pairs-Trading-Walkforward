# src/signals.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalConfig:
    """
    Signal generation configuration.

    Learning notes:
    - z_lookback should be long enough to estimate mean/std (e.g. 60 trading days).
    - entry_z / exit_z define a simple hysteresis band to avoid churn.
    - hard_stop_z is a safety feature (optional) to cut runaway spreads.
    - max_holding_days is a time stop (optional) to avoid holding forever.
    """
    z_lookback: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    hard_stop_z: Optional[float] = None        # e.g. 4.0, or None to disable
    max_holding_days: Optional[int] = None     # e.g. 20, or None to disable


def compute_spread(
    prices: pd.DataFrame,
    pair: Tuple[str, str],
    alpha: float,
    beta: float,
) -> pd.Series:
    """
    Spread (residual) defined as:
        spread_t = log(A_t) - (alpha + beta * log(B_t))

    alpha/beta must be estimated on the TRAIN window and held fixed out-of-sample.
    """
    a, b = pair
    if a not in prices.columns or b not in prices.columns:
        raise KeyError(f"Pair tickers {pair} not both present in prices columns.")

    if (prices[[a, b]] <= 0).any().any():
        raise ValueError("Prices must be positive to compute log prices.")

    logA = np.log(prices[a].astype(float))
    logB = np.log(prices[b].astype(float))
    spread = logA - (alpha + beta * logB)
    spread.name = f"spread_{a}_{b}"
    return spread


def compute_zscore(spread: pd.Series, lookback: int) -> pd.Series:
    """
    Rolling z-score:
        z_t = (spread_t - mean_{t-lookback:t-1}) / std_{t-lookback:t-1}

    Implementation detail:
    - rolling mean/std computed including current point by default,
      but we shift by 1 to ensure we use past-only statistics.
    """
    rolling_mean = spread.rolling(lookback).mean().shift(1)
    rolling_std = spread.rolling(lookback).std(ddof=1).shift(1)
    z = (spread - rolling_mean) / rolling_std
    z.name = "zscore"
    return z


def generate_positions_from_zscore(
    z: pd.Series,
    cfg: SignalConfig,
) -> pd.Series:
    """
    Generate discrete positions:
      +1 : long spread  (long A, short B)
      -1 : short spread (short A, long B)
       0 : flat

    Rules (hysteresis):
      - If flat:
          enter short if z > entry_z
          enter long  if z < -entry_z
      - If long:
          exit if z > -exit_z (i.e. z crosses back toward 0)
      - If short:
          exit if z < exit_z

    Optional:
      - hard_stop_z: exit if |z| exceeds hard_stop_z
      - max_holding_days: exit after N days in a position
    """
    pos = np.zeros(len(z), dtype=int)
    holding = 0  # days in current position

    for i in range(len(z)):
        zi = z.iat[i]

        # If z is nan (early periods), stay flat
        if not np.isfinite(zi):
            pos[i] = 0
            holding = 0
            continue

        prev = pos[i - 1] if i > 0 else 0
        current = prev

        # Track holding duration
        if current != 0:
            holding += 1
        else:
            holding = 0

        # Optional hard stop: cut any position if spread is too extreme
        if cfg.hard_stop_z is not None and abs(zi) >= cfg.hard_stop_z:
            current = 0
            holding = 0

        # Optional time stop
        if cfg.max_holding_days is not None and holding >= cfg.max_holding_days:
            current = 0
            holding = 0

        # Main hysteresis logic (only if we didn't already exit via stops)
        if current == 0:
            if zi > cfg.entry_z:
                current = -1  # short spread
                holding = 0
            elif zi < -cfg.entry_z:
                current = +1  # long spread
                holding = 0
        elif current == +1:
            # Exit long when z has reverted back upward toward 0
            if zi >= -cfg.exit_z:
                current = 0
                holding = 0
        elif current == -1:
            # Exit short when z has reverted back downward toward 0
            if zi <= cfg.exit_z:
                current = 0
                holding = 0

        pos[i] = current

    positions = pd.Series(pos, index=z.index, name="position")
    return positions


def generate_signals(
    prices: pd.DataFrame,
    pair: Tuple[str, str],
    alpha: float,
    beta: float,
    cfg: SignalConfig = SignalConfig(),
) -> pd.DataFrame:
    """
    Build a signal dataframe with:
      - spread
      - zscore (past-only rolling stats)
      - position (+1 long spread, -1 short spread, 0 flat)

    This output is designed to feed directly into your backtester.
    """
    spread = compute_spread(prices, pair, alpha, beta)
    z = compute_zscore(spread, cfg.z_lookback)
    position = generate_positions_from_zscore(z, cfg)

    out = pd.concat([spread, z, position], axis=1)
    return out
