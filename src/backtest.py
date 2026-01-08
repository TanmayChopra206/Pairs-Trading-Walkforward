
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    """
    Backtest settings.

    Learning notes:
    - cost_bps applies per unit turnover (rough model of spread+slippage).
      Example: 10 bps = 0.001 per $1 notional traded.
    - We backtest a *spread portfolio*:
        long A, short beta * B when position = +1
        short A, long beta * B when position = -1
      This is not dollar-neutral by construction unless you normalize notionals;
      we normalise by (1 + |beta|) to make returns comparable across pairs.
    """
    cost_bps: float = 10.0         # 10 bps per unit turnover (both legs combined approx)
    annualisation: int = 252


def _log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    if (prices <= 0).any().any():
        raise ValueError("Prices must be positive for log returns.")
    return np.log(prices.astype(float)).diff().dropna()


def compute_spread_returns(
    prices: pd.DataFrame,
    pair: Tuple[str, str],
    beta: float,
) -> pd.Series:
    """
    Compute daily returns of the spread portfolio, before applying position.

    We approximate a spread portfolio return using log returns:
      r_spread = r_A - beta * r_B

    This is a common approximation for pairs trading with log-price modelling.
    We then normalise by (1 + |beta|) to keep return magnitudes comparable.
    """
    a, b = pair
    rets = _log_returns(prices[[a, b]])
    r = rets[a] - beta * rets[b]
    r = r / (1.0 + abs(beta))
    r.name = "spread_ret"
    return r


def compute_turnover(position: pd.Series) -> pd.Series:
    """
    Turnover proxy based on changes in position.
    position in {-1,0,+1}. Turnover is |Δpos|.
    """
    pos = position.fillna(0).astype(int)
    turnover = pos.diff().abs().fillna(0.0)
    turnover.name = "turnover"
    return turnover


def backtest_pair(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    pair: Tuple[str, str],
    beta: float,
    cfg: BacktestConfig = BacktestConfig(),
) -> pd.DataFrame:
    """
    Run a single-pair backtest.

    Inputs:
    - prices: DataFrame with columns including pair tickers
    - signals: output of generate_signals(), indexed by date, must include 'position'
    - pair: (A, B)
    - beta: hedge ratio (from training)
    - cfg: cost + annualisation settings

    Output DataFrame columns:
    - spread_ret: raw spread return (no position)
    - position: desired position (+1/-1/0)
    - position_lag: position shifted by 1 day (execution lag)
    - gross_ret: position_lag * spread_ret
    - turnover: |Δposition_lag|
    - cost: turnover * cost_rate
    - net_ret: gross_ret - cost
    - equity: cumulative wealth (starting at 1)
    """
    a, b = pair

    # Align everything on common dates
    prices_sub = prices[[a, b]].copy()
    signals = signals.copy()

    # spread returns available from prices
    spread_ret = compute_spread_returns(prices_sub, pair, beta)

    # align signals index to spread_ret index
    sig = signals.reindex(spread_ret.index).copy()
    if "position" not in sig.columns:
        raise KeyError("signals must contain a 'position' column.")

    position = sig["position"].fillna(0).astype(int)

    # No lookahead: trade on next bar
    position_lag = position.shift(1).fillna(0).astype(int)
    position_lag.name = "position_lag"

    gross_ret = position_lag * spread_ret
    gross_ret.name = "gross_ret"

    turnover = compute_turnover(position_lag)

    # Convert bps to decimal cost rate
    cost_rate = cfg.cost_bps / 10_000.0
    cost = turnover * cost_rate
    cost.name = "cost"

    net_ret = gross_ret - cost
    net_ret.name = "net_ret"

    equity = (1.0 + net_ret).cumprod()
    equity.name = "equity"

    out = pd.concat(
        [spread_ret, position, position_lag, gross_ret, turnover, cost, net_ret, equity],
        axis=1
    )

    return out


def performance_summary(
    net_returns: pd.Series,
    cfg: BacktestConfig = BacktestConfig(),
) -> Dict[str, float]:
    """
    Compute basic performance metrics for strategy net returns.
    """
    r = net_returns.dropna().astype(float)
    if len(r) == 0:
        return {
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "avg_daily_ret": np.nan,
            "vol_daily": np.nan,
            "hit_rate": np.nan,
            "n_days": 0,
        }

    ann = cfg.annualisation
    avg = r.mean()
    vol = r.std(ddof=1)

    ann_return = (1.0 + avg) ** ann - 1.0
    ann_vol = vol * np.sqrt(ann)
    sharpe = (avg / vol) * np.sqrt(ann) if vol > 0 else np.nan

    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    max_dd = dd.min()

    hit_rate = float((r > 0).mean())

    return {
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "max_drawdown": float(max_dd),
        "avg_daily_ret": float(avg),
        "vol_daily": float(vol),
        "hit_rate": float(hit_rate),
        "n_days": int(len(r)),
    }


def extract_trades(position: pd.Series) -> pd.DataFrame:
    """
    Very simple trade extraction from a position series in {-1,0,+1}.
    Returns a dataframe with entry/exit dates and direction.
    """
    pos = position.fillna(0).astype(int)
    changes = pos.diff().fillna(pos)

    entries = pos[(changes != 0) & (pos != 0)]
    exits = pos[(changes != 0) & (pos == 0)]

    trades = []
    open_trade = None

    for dt, p in entries.items():
        if open_trade is None:
            open_trade = {"entry_date": dt, "direction": int(p)}
        else:
            # If we enter while already in a position, close previous and open new
            trades.append({**open_trade, "exit_date": dt})
            open_trade = {"entry_date": dt, "direction": int(p)}

    for dt in exits.index:
        if open_trade is not None and dt > open_trade["entry_date"]:
            trades.append({**open_trade, "exit_date": dt})
            open_trade = None

    return pd.DataFrame(trades)
