from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import pandas as pd
from dateutil.relativedelta import relativedelta


@dataclass(frozen=True)
class WalkForwardConfig:
    train_years: int = 2
    test_months: int = 6
    step_months: int = 6


def generate_walkforward_splits(
    dates: pd.DatetimeIndex,
    cfg: WalkForwardConfig = WalkForwardConfig(),
) -> Iterator[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Generate walk-forward (train_start, train_end, test_start, test_end) tuples.
    Uses calendar deltas, then snaps to available dates in `dates`.
    """
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.to_datetime(dates)

    dates = dates.sort_values().unique()
    start = pd.Timestamp(dates[0])
    end = pd.Timestamp(dates[-1])

    train_start = start
    train_end = train_start + relativedelta(years=cfg.train_years) - pd.Timedelta(days=1)

    while True:
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + relativedelta(months=cfg.test_months) - pd.Timedelta(days=1)

        if test_end > end:
            break

        # Snap to actual available dates
        tr_start = _snap_forward(dates, train_start)
        tr_end = _snap_backward(dates, train_end)
        te_start = _snap_forward(dates, test_start)
        te_end = _snap_backward(dates, test_end)

        # Ensure ordering is valid
        if tr_start < tr_end and te_start < te_end and tr_end < te_start:
            yield tr_start, tr_end, te_start, te_end

        # Step forward
        train_end = train_end + relativedelta(months=cfg.step_months)
        # train_start stays fixed expanding-window; if you want rolling-window, also move train_start forward.

        if train_end >= end:
            break


def _snap_forward(dates: pd.DatetimeIndex, t: pd.Timestamp) -> pd.Timestamp:
    idx = dates.searchsorted(t, side="left")
    if idx >= len(dates):
        return pd.Timestamp(dates[-1])
    return pd.Timestamp(dates[idx])


def _snap_backward(dates: pd.DatetimeIndex, t: pd.Timestamp) -> pd.Timestamp:
    idx = dates.searchsorted(t, side="right") - 1
    if idx < 0:
        return pd.Timestamp(dates[0])
    return pd.Timestamp(dates[idx])
