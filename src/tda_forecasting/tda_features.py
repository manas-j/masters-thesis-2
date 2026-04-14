from __future__ import annotations

import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from persim import PersistenceImager
from tqdm import tqdm
from ripser import ripser
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity


@dataclass(slots=True)
class TDAFeatureConfig:
    tau: int = 1
    embed_dim: int = 3
    window_len: int = 60
    kde_bandwidth: float = 0.5
    density_quantile: float = 0.05
    maxdim: int = 2
    use_density_filter: bool = True


# Populated by ProcessPool initializer (chunk workers).
_MP_X: np.ndarray | None = None
_MP_WINDOW: int = 0
_MP_CFG: TDAFeatureConfig | None = None
_MP_COLS: np.ndarray | None = None  # None = all columns; else column index array


def _mp_init_worker(x_arr: np.ndarray, window_len: int, cfg_dict: dict, col_indices: np.ndarray | None) -> None:
    global _MP_X, _MP_WINDOW, _MP_CFG, _MP_COLS
    _MP_X = x_arr
    _MP_WINDOW = int(window_len)
    _MP_CFG = TDAFeatureConfig(**cfg_dict)
    _MP_COLS = col_indices


def _mp_process_chunk(ts: list[int]) -> list[dict[str, float]]:
    assert _MP_X is not None and _MP_CFG is not None
    wlen = _MP_WINDOW
    rows: list[dict[str, float]] = []
    for t in ts:
        sl = _MP_X[t - wlen + 1 : t + 1]
        if _MP_COLS is not None:
            sl = np.ascontiguousarray(sl[:, _MP_COLS])
        rows.append(extract_window_tda_features(pd.DataFrame(sl), _MP_CFG))
    return rows


def takens_embedding(X: np.ndarray, tau: int, embed_dim: int) -> np.ndarray:
    n, p = X.shape
    m = n - (embed_dim - 1) * tau
    if m <= 5:
        raise ValueError("Embedding too short. Increase window or reduce tau/embed_dim.")

    parts = [X[np.arange(m, dtype=np.intp) + k * tau] for k in range(embed_dim)]
    return np.concatenate(parts, axis=1)


def _density_mask(points: np.ndarray, bandwidth: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    kde = KernelDensity(kernel="gaussian", bandwidth=max(bandwidth, 1e-6))
    kde.fit(points)
    log_density = kde.score_samples(points)
    thr = np.quantile(log_density, q)
    keep = log_density >= thr
    return keep, log_density


def _diagram_rows_for_stats(dgm: np.ndarray, hom_dim: int) -> np.ndarray:
    """
    ripser marks the essential H0 class with death = inf. Dropping those rows
    (via an all-finite mask) removes the longest-lived component and flattens
    H0 entropy / amplitude / counts. Impute inf death with a scale derived from
    finite merge radii so statistics remain comparable across windows.
    """
    if dgm.size == 0:
        return dgm.reshape(0, 2)
    dgm = np.asarray(dgm, dtype=np.float64).copy()
    births, deaths = dgm[:, 0], dgm[:, 1]
    finite_death = np.isfinite(deaths)
    if hom_dim == 0:
        if np.any(finite_death):
            cap = float(np.nanmax(deaths[finite_death]))
        else:
            cap = float(np.nanmax(births)) if births.size else 1.0
        cap = max(cap, 1e-12)
        deaths_imputed = deaths.copy()
        deaths_imputed[~finite_death] = cap * 1.25
        dgm[:, 1] = deaths_imputed
        valid = np.isfinite(dgm).all(axis=1) & (dgm[:, 1] >= dgm[:, 0])
        return dgm[valid]
    return dgm[np.isfinite(dgm).all(axis=1)]


def _pd_stats(diag: np.ndarray) -> tuple[float, float, float]:
    if diag.size == 0:
        return 0.0, 0.0, 0.0
    life = np.maximum(diag[:, 1] - diag[:, 0], 1e-12)
    p = life / np.sum(life)
    entropy = float(-np.sum(p * np.log(p)))
    amplitude = float(np.max(life))
    n_points = float(diag.shape[0])
    return entropy, amplitude, n_points


def _stabilize_point_cloud(points: np.ndarray) -> np.ndarray:
    """
    Joint Takens vectors live in R^{D * embed_dim}; for multivariate D this often
    exceeds the number of window points. KDE and Vietoris–Rips are unreliable in
    that regime, so we PCA-compress using only the current window (segment-wise).
    """
    n, d = points.shape
    if n <= 4:
        return points
    max_allowed = max(2, n - 3)
    if d <= max_allowed:
        return points
    return PCA(n_components=max_allowed, svd_solver="full", random_state=0).fit_transform(points)


def filtered_embedding_points(window_df: pd.DataFrame, cfg: TDAFeatureConfig) -> tuple[np.ndarray, dict[str, float]]:
    """Takens embed, optional PCA stabilization, then KDE hard-threshold (see thesis Ch.4)."""
    points = takens_embedding(window_df.to_numpy(dtype=np.float64), tau=cfg.tau, embed_dim=cfg.embed_dim)
    points = _stabilize_point_cloud(points)
    meta: dict[str, float] = {}
    if cfg.use_density_filter:
        keep, log_density = _density_mask(points, bandwidth=cfg.kde_bandwidth, q=cfg.density_quantile)
        filtered = points[keep]
        meta["density_mean"] = float(np.mean(np.exp(np.clip(log_density, -50.0, 50.0))))
        meta["density_kept_ratio"] = float(np.mean(keep))
    else:
        filtered = points
        meta["density_mean"] = 0.0
        meta["density_kept_ratio"] = 1.0
    if filtered.shape[0] < 8:
        filtered = points
    filtered = _stabilize_point_cloud(filtered)
    return filtered, meta


def extract_window_tda_features(window_df: pd.DataFrame, cfg: TDAFeatureConfig) -> dict[str, float]:
    filtered_points, dens_meta = filtered_embedding_points(window_df, cfg)
    feats: dict[str, float] = dict(dens_meta)

    dgms = ripser(filtered_points, maxdim=cfg.maxdim)["dgms"]

    for dim in range(cfg.maxdim + 1):
        dgm = dgms[dim]
        if dgm.size:
            dgm = _diagram_rows_for_stats(dgm, hom_dim=dim)
        ent, amp, npts = _pd_stats(dgm)
        feats[f"H{dim}_entropy"] = ent
        feats[f"H{dim}_amplitude"] = amp
        feats[f"H{dim}_npoints"] = npts

    dgm1 = dgms[1]
    if dgm1.size:
        dgm1 = _diagram_rows_for_stats(dgm1, hom_dim=1)
        if dgm1.size:
            spread = float(max(np.ptp(dgm1[:, 0]), np.ptp(dgm1[:, 1]), 1e-6))
            pixel_size = max(0.05 * spread, 1e-4)
            pimgr = PersistenceImager(pixel_size=pixel_size)
            pimgr.fit(dgm1)
            img = np.asarray(pimgr.transform(dgm1))
            if img.size == 0:
                feats["PI_H1_sum"] = 0.0
                feats["PI_H1_max"] = 0.0
            else:
                feats["PI_H1_sum"] = float(np.sum(img))
                feats["PI_H1_max"] = float(np.max(img))
        else:
            feats["PI_H1_sum"] = 0.0
            feats["PI_H1_max"] = 0.0
    else:
        feats["PI_H1_sum"] = 0.0
        feats["PI_H1_max"] = 0.0

    return feats


def _resolve_tda_workers(num_windows: int) -> int:
    raw = os.environ.get("TDA_FORECASTING_TDA_WORKERS", "").strip()
    if raw:
        return max(1, int(raw))
    if num_windows < 48:
        return 1
    cpu = os.cpu_count() or 4
    return max(1, min(8, cpu))


def _process_pool_ctx() -> mp.context.BaseContext:
    # Avoid forkserver (Py3.12+) re-importing __main__ as "<stdin>" in notebooks / python -c.
    if sys.platform.startswith("linux"):
        return mp.get_context("fork")
    return mp.get_context("spawn")


def build_feature_matrix(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: TDAFeatureConfig,
    use_univariate_only: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    One-step-ahead targets: at forecast origin t (last day in the window), predict y[t+1].
    Window rows are X[t-L+1 : t+1] inclusive (length L = window_len).
    """
    window = cfg.window_len
    if window < 2:
        raise ValueError("window_len must be at least 2.")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length.")

    X_arr = np.ascontiguousarray(X.to_numpy(dtype=np.float64, copy=False))
    ys = y.reset_index(drop=True)
    m5 = ys.rolling(5, min_periods=1).mean().to_numpy(dtype=np.float64, copy=False)
    s5 = ys.rolling(5, min_periods=1).std(ddof=0).to_numpy(dtype=np.float64, copy=False)
    s5 = np.nan_to_num(s5, nan=0.0)
    m20 = ys.rolling(20, min_periods=1).mean().to_numpy(dtype=np.float64, copy=False)

    ts = list(range(window - 1, len(X) - 1))
    if not ts:
        return pd.DataFrame(), pd.Series(dtype=float, name="target")

    col_indices: np.ndarray | None = None
    if use_univariate_only:
        if y.name not in X.columns:
            raise KeyError(f"Target series name {y.name!r} not in feature columns.")
        loc = X.columns.get_loc(y.name)
        if not isinstance(loc, (int, np.integer)):
            raise TypeError("Target column must resolve to a single integer index for univariate-only mode.")
        col_indices = np.array([int(loc)], dtype=np.intp)

    workers = _resolve_tda_workers(len(ts))
    cfg_dict = asdict(cfg)

    if workers <= 1:
        tda_rows: list[dict[str, float]] = []
        for t in tqdm(ts, desc="TDA windows", unit="win", leave=False):
            sl = X_arr[t - window + 1 : t + 1]
            if col_indices is not None:
                sl = np.ascontiguousarray(sl[:, col_indices])
            tda_rows.append(extract_window_tda_features(pd.DataFrame(sl), cfg))
    else:
        n_chunks = min(workers, len(ts))
        chunks = [c.tolist() for c in np.array_split(np.asarray(ts, dtype=np.int64), n_chunks) if len(c) > 0]
        with ProcessPoolExecutor(
            max_workers=len(chunks),
            mp_context=_process_pool_ctx(),
            initializer=_mp_init_worker,
            initargs=(X_arr, window, cfg_dict, col_indices),
        ) as ex:
            parts = list(
                tqdm(
                    ex.map(_mp_process_chunk, chunks),
                    total=len(chunks),
                    desc="TDA chunks",
                    unit="chunk",
                    leave=False,
                )
            )
        tda_rows = [row for part in parts for row in part]

    records: list[dict[str, float]] = []
    targets: list[float] = []
    for t, tda in zip(ts, tda_rows, strict=True):
        lag_feats = {
            "lag_mean_5": float(m5[t]),
            "lag_std_5": float(s5[t]),
            "lag_mean_20": float(m20[t]),
        }
        records.append({**lag_feats, **tda})
        targets.append(float(ys.iloc[t + 1]))

    return pd.DataFrame.from_records(records), pd.Series(targets, name="target")
