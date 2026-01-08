from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class DataConfig:
    """
    Configuration for downloading + cleaning price data.

    Learning note:
    - max_missing_frac controls how strict you are about incomplete tickers.
      In trading, missing data can create fake signals and hidden biases.
    - align_common_start ensures every ticker has data from the same start date,
      which prevents weird comparisons early in the sample.
    """
    start: str = "2015-01-01"
    end: Optional[str] = None
    price_field: str = "Adj Close"
    cache_dir: str = "data/cache"
    cache_name: str = "prices_adjclose.parquet"
    max_missing_frac: float = 0.02 
    align_common_start: bool = True
    drop_any_remaining_na_rows: bool = True


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def download_prices(
    tickers: Iterable[str],
    cfg: DataConfig = DataConfig(),
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download and return a clean price matrix (wide DataFrame).

    Returns:
        prices_df:
            index: DatetimeIndex
            columns: tickers
            values: adjusted close prices (float)

    How to use:
        prices = download_prices(TICKERS, DataConfig(cache_name="us_banks.parquet"))
    """
    tickers = [t.strip().upper() for t in tickers]
    tickers = list(dict.fromkeys(tickers))  # dedupe while preserving order

    cache_dir = Path(cfg.cache_dir)
    _ensure_dir(cache_dir)
    cache_path = cache_dir / cfg.cache_name

    # --- Load from cache if available ---
    if cache_path.exists() and not force_refresh:
        prices = pd.read_parquet(cache_path)
        prices.index = pd.to_datetime(prices.index)
        return clean_prices(prices, cfg)

    # --- Download from yfinance ---
    raw = yf.download(
        tickers=tickers,
        start=cfg.start,
        end=cfg.end,
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    # Extract the desired price field robustly
    if isinstance(raw.columns, pd.MultiIndex):
        # Typical case for multiple tickers: columns like ('Adj Close', 'JPM')
        if cfg.price_field not in raw.columns.get_level_values(0):
            raise ValueError(f"'{cfg.price_field}' not found in yfinance response.")
        prices = raw[cfg.price_field].copy()
    else:
        # Single ticker case (rare for your use), columns like 'Adj Close'
        if cfg.price_field not in raw.columns:
            raise ValueError(f"'{cfg.price_field}' not found in yfinance response.")
        prices = raw[[cfg.price_field]].copy()
        prices.columns = [tickers[0]]

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().astype("float64")

    # Cache the extracted prices (before cleaning) so you can inspect if needed
    prices.to_parquet(cache_path)

    return clean_prices(prices, cfg)


def clean_prices(prices: pd.DataFrame, cfg: DataConfig = DataConfig()) -> pd.DataFrame:
    """
    Clean a raw price matrix.

    Cleaning steps:
    1) Drop all-null columns.
    2) Drop tickers with missingness above threshold.
    3) Optionally align to a common start date across remaining tickers.
    4) Optionally drop any remaining NA rows.
    5) Sanity check for non-positive prices.
    """
    prices = prices.copy()
    prices = prices.loc[~prices.index.duplicated(keep="first")]
    prices = prices.sort_index()

    # 1) Drop columns that are entirely NA
    prices = prices.dropna(axis=1, how="all")

    # 2) Drop tickers with too much missingness
    miss_frac = prices.isna().mean(axis=0)
    keep_cols = miss_frac[miss_frac <= cfg.max_missing_frac].index.tolist()
    dropped = sorted(set(prices.columns) - set(keep_cols))
    if dropped:
        print(f"[data] Dropping tickers (missingness > {cfg.max_missing_frac:.0%}): {dropped}")
    prices = prices[keep_cols]

    if prices.shape[1] < 2:
        raise ValueError("Not enough tickers left after cleaning. Relax filters or change universe.")

    # 3) Align to common start date (so all tickers have valid data from same point)
    if cfg.align_common_start:
        first_valid = []
        for c in prices.columns:
            idx = prices[c].first_valid_index()
            if idx is not None:
                first_valid.append(idx)
        common_start = max(first_valid)
        prices = prices.loc[prices.index >= common_start]

    # 4) Drop any remaining NA rows (strict but safe for research)
    if cfg.drop_any_remaining_na_rows:
        prices = prices.dropna(axis=0, how="any")

    # 5) Sanity check: prices should be positive
    if (prices <= 0).any().any():
        raise ValueError("Found non-positive prices after cleaning. Investigate data integrity.")

    return prices


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns from prices.

    Why log returns?
    - Additive over time
    - Better behaved for many statistical operations
    """
    return np.log(prices).diff().dropna()


def sanity_report(prices: pd.DataFrame) -> dict:
    """
    Quick sanity stats for learning/debugging.
    """
    report = {
        "rows": int(prices.shape[0]),
        "cols": int(prices.shape[1]),
        "start": str(prices.index.min().date()),
        "end": str(prices.index.max().date()),
        "tickers": list(prices.columns),
        "missing_frac_max": float(prices.isna().mean().max()),
    }
    return report
