from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from persim import PersistenceImager
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


def _standardize_window(window: pd.DataFrame) -> np.ndarray:
    arr = window.to_numpy(dtype=float)
    mu = arr.mean(axis=0, keepdims=True)
    sigma = arr.std(axis=0, keepdims=True)
    sigma[sigma < 1e-12] = 1.0
    return (arr - mu) / sigma


def takens_embedding(X: np.ndarray, tau: int, embed_dim: int) -> np.ndarray:
    n, p = X.shape
    m = n - (embed_dim - 1) * tau
    if m <= 5:
        raise ValueError("Embedding too short. Increase window or reduce tau/embed_dim.")

    emb = []
    for i in range(m):
        row = [X[i + k * tau] for k in range(embed_dim)]
        emb.append(np.concatenate(row))
    return np.asarray(emb)


def _density_mask(points: np.ndarray, bandwidth: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(points)
    log_density = kde.score_samples(points)
    thr = np.quantile(log_density, q)
    keep = log_density >= thr
    return keep, log_density


def _density_weighted_points(points: np.ndarray, log_density: np.ndarray) -> np.ndarray:
    density = np.exp(log_density)
    # Down-weight sparse/noisy regions by shrinking points with low density.
    scale = density / (np.median(density) + 1e-12)
    scale = np.clip(scale, 0.25, 2.0)[:, None]
    return points * scale


def _prepare_points_for_ph(points: np.ndarray) -> np.ndarray:
    n_rows, n_cols = points.shape
    if n_cols < n_rows:
        return points
    # Stabilize Vietoris-Rips when embedding dimensionality is very high.
    n_components = max(3, min(n_rows - 1, 32))
    return PCA(n_components=n_components).fit_transform(points)


def _pd_stats(diag: np.ndarray) -> tuple[float, float, float]:
    if diag.size == 0:
        return 0.0, 0.0, 0.0
    life = np.maximum(diag[:, 1] - diag[:, 0], 1e-12)
    p = life / np.sum(life)
    entropy = float(-np.sum(p * np.log(p)))
    amplitude = float(np.max(life))
    n_points = float(diag.shape[0])
    return entropy, amplitude, n_points


def extract_window_tda_features(window_df: pd.DataFrame, cfg: TDAFeatureConfig) -> dict[str, float]:
    points = takens_embedding(_standardize_window(window_df), tau=cfg.tau, embed_dim=cfg.embed_dim)
    feats: dict[str, float] = {}

    if cfg.use_density_filter:
        keep, log_density = _density_mask(points, bandwidth=cfg.kde_bandwidth, q=cfg.density_quantile)
        filtered_points = _density_weighted_points(points[keep], log_density[keep])
        feats["density_mean"] = float(np.mean(np.exp(log_density)))
        feats["density_kept_ratio"] = float(np.mean(keep))
    else:
        filtered_points = points
        feats["density_mean"] = 0.0
        feats["density_kept_ratio"] = 1.0

    if filtered_points.shape[0] < 8:
        filtered_points = points

    ph_points = _prepare_points_for_ph(filtered_points)
    dgms = ripser(ph_points, maxdim=cfg.maxdim)["dgms"]

    for dim in range(cfg.maxdim + 1):
        dgm = dgms[dim]
        if dgm.size:
            dgm = dgm[np.isfinite(dgm).all(axis=1)]
        ent, amp, npts = _pd_stats(dgm)
        feats[f"H{dim}_entropy"] = ent
        feats[f"H{dim}_amplitude"] = amp
        feats[f"H{dim}_npoints"] = npts

    if dgms[1].size:
        dgm1 = dgms[1][np.isfinite(dgms[1]).all(axis=1)]
        if dgm1.size:
            pimgr = PersistenceImager(pixel_size=0.2)
            pimgr.fit(dgm1)
            img = pimgr.transform(dgm1)
            feats["PI_H1_sum"] = float(np.sum(img)) if img.size else 0.0
            feats["PI_H1_max"] = float(np.max(img)) if img.size else 0.0
        else:
            feats["PI_H1_sum"] = 0.0
            feats["PI_H1_max"] = 0.0
    else:
        feats["PI_H1_sum"] = 0.0
        feats["PI_H1_max"] = 0.0

    return feats


def build_feature_matrix(X: pd.DataFrame, y: pd.Series, cfg: TDAFeatureConfig, use_univariate_only: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    window = cfg.window_len
    records: list[dict[str, float]] = []
    targets: list[float] = []

    for end in range(window, len(X) - 1):
        w = X.iloc[end - window : end]
        if use_univariate_only:
            w = w[[y.name]]

        lag_feats = {}
        target_slice = y.iloc[end - 20 : end]
        lag_feats["lag_mean_5"] = float(target_slice.iloc[-5:].mean())
        lag_feats["lag_std_5"] = float(target_slice.iloc[-5:].std(ddof=0))
        lag_feats["lag_mean_20"] = float(target_slice.mean())

        tda = extract_window_tda_features(w, cfg)
        row = {**lag_feats, **tda}
        records.append(row)
        targets.append(float(y.iloc[end + 1]))

    return pd.DataFrame.from_records(records), pd.Series(targets, name="target")
