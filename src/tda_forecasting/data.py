from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(slots=True)
class DataBundle:
    X: pd.DataFrame
    y: pd.Series


def _safe_log_return(series: pd.Series) -> pd.Series:
    return np.log(series).diff().replace([np.inf, -np.inf], np.nan)


def _safe_log1p_diff(series: pd.Series) -> pd.Series:
    return np.log1p(series.clip(lower=0)).diff().replace([np.inf, -np.inf], np.nan)


def load_market_data(start: str, end: str, target_ticker: str, feature_tickers: list[str]) -> DataBundle:
    cache_dir = Path(".cache/yfinance")
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir.resolve()))

    tickers = sorted(set([target_ticker] + feature_tickers))
    raw = yf.download(tickers=tickers, start=start, end=end, auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError("No data downloaded; check tickers/date range.")

    # Yahoo returns MultiIndex columns when multiple tickers are requested.
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"] if "Close" in raw else pd.DataFrame(index=raw.index)
        open_ = raw["Open"] if "Open" in raw else pd.DataFrame(index=raw.index)
        high = raw["High"] if "High" in raw else pd.DataFrame(index=raw.index)
        low = raw["Low"] if "Low" in raw else pd.DataFrame(index=raw.index)
        volume = raw["Volume"] if "Volume" in raw else pd.DataFrame(index=raw.index)
    else:
        close = raw[["Close"]].rename(columns={"Close": target_ticker})
        open_ = raw[["Open"]].rename(columns={"Open": target_ticker})
        high = raw[["High"]].rename(columns={"High": target_ticker})
        low = raw[["Low"]].rename(columns={"Low": target_ticker})
        volume = raw[["Volume"]].rename(columns={"Volume": target_ticker})

    feats = pd.DataFrame(index=close.index)
    for tk in tickers:
        if tk in close:
            feats[f"{tk}_ret"] = _safe_log_return(close[tk])
        if tk in open_:
            feats[f"{tk}_oc_ret"] = np.log(close[tk] / open_[tk]).replace([np.inf, -np.inf], np.nan)
        if tk in high and tk in low:
            feats[f"{tk}_hl_spread"] = np.log(high[tk] / low[tk]).replace([np.inf, -np.inf], np.nan)
        if tk in volume:
            feats[f"{tk}_vol_chg"] = _safe_log1p_diff(volume[tk])

    feats = feats.interpolate(method="time").ffill().dropna()
    target = feats[f"{target_ticker}_ret"].copy()
    return DataBundle(X=feats, y=target)
