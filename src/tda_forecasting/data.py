from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class DataBundle:
    X: pd.DataFrame
    y: pd.Series


def _safe_log_return(series: pd.Series) -> pd.Series:
    return np.log(series).diff().replace([np.inf, -np.inf], np.nan)


def _series_for_ticker(raw: pd.DataFrame, field: str, ticker: str) -> pd.Series | None:
    if isinstance(raw.columns, pd.MultiIndex):
        key = (field, ticker)
        if key not in raw.columns:
            return None
        return raw[key]
    if field not in raw.columns:
        return None
    return raw[field]


def load_market_data(start: str, end: str, target_ticker: str, feature_tickers: list[str]) -> DataBundle:
    tickers = sorted(set([target_ticker] + feature_tickers))
    raw = yf.download(tickers=tickers, start=start, end=end, auto_adjust=False, progress=False, group_by="column")
    if raw.empty:
        raise RuntimeError("No data downloaded; check tickers/date range.")

    feats = pd.DataFrame(index=raw.index)

    for tk in tickers:
        close = _series_for_ticker(raw, "Close", tk)
        if close is not None and not close.isna().all():
            feats[f"{tk}_ret"] = _safe_log_return(close)
        for field, suffix in (("Open", "open_ret"), ("High", "high_ret"), ("Low", "low_ret")):
            s = _series_for_ticker(raw, field, tk)
            if s is not None and not s.isna().all():
                feats[f"{tk}_{suffix}"] = _safe_log_return(s)

        vol = _series_for_ticker(raw, "Volume", tk)
        if vol is not None and not vol.isna().all():
            feats[f"{tk}_vol_chg"] = np.log1p(vol.clip(lower=0)).diff()

    feats = feats.interpolate(method="time").ffill().bfill().dropna(how="all")
    feats = feats.dropna()
    if feats.empty:
        raise RuntimeError("Feature matrix empty after cleaning; check tickers/date range.")

    target_col = f"{target_ticker}_ret"
    if target_col not in feats.columns:
        raise RuntimeError(f"Missing target column {target_col} in downloaded data.")
    target = feats[target_col].copy()
    return DataBundle(X=feats, y=target)


def standardize_features_train_stats(X: pd.DataFrame, train_end_exclusive: int) -> pd.DataFrame:
    """Standardize columns using mean/variance from X.iloc[:train_end_exclusive] only (causal)."""
    train_end_exclusive = max(1, min(train_end_exclusive, len(X)))
    scaler = StandardScaler()
    scaler.fit(X.iloc[:train_end_exclusive].to_numpy())
    scaled = scaler.transform(X.to_numpy())
    return pd.DataFrame(scaled, index=X.index, columns=X.columns)
