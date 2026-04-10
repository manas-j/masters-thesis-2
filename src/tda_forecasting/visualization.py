from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

from .config import ExperimentConfig
from .data import load_market_data
from .tda_features import TDAFeatureConfig, build_feature_matrix, takens_embedding


def _pca_2d(points: np.ndarray) -> np.ndarray:
    if points.shape[1] <= 2:
        return points[:, :2]
    return PCA(n_components=2).fit_transform(points)


def _kde_keep_mask(points: np.ndarray, bandwidth: float, quantile: float) -> np.ndarray:
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(points)
    log_density = kde.score_samples(points)
    threshold = np.quantile(log_density, quantile)
    return log_density >= threshold


def _prepare_data(cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.Series]:
    try:
        bundle = load_market_data(
            start=cfg.data["start"],
            end=cfg.data["end"],
            target_ticker=cfg.data["target_ticker"],
            feature_tickers=cfg.data["feature_tickers"],
        )
        return bundle.X, bundle.y
    except Exception as exc:
        warnings.warn(
            f"Falling back to synthetic visualization data because market data download failed: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        n = 1200
        idx = pd.date_range("2016-01-01", periods=n, freq="D")
        t = np.linspace(0, 24 * np.pi, n)
        rng = np.random.default_rng(42)
        X = pd.DataFrame(
            {
                "synthetic_ret": 0.015 * np.sin(t) + 0.01 * rng.normal(size=n),
                "synthetic_oc_ret": 0.012 * np.cos(1.3 * t) + 0.01 * rng.normal(size=n),
                "synthetic_hl_spread": 0.03 + 0.01 * np.sin(0.5 * t) + 0.005 * rng.normal(size=n),
                "synthetic_vol_chg": 0.02 * np.sin(0.2 * t + 0.5) + 0.01 * rng.normal(size=n),
            },
            index=idx,
        )
        y = X["synthetic_ret"].copy()
        return X, y


def _robust_tda_cfg(cfg: ExperimentConfig) -> TDAFeatureConfig:
    return TDAFeatureConfig(
        tau=cfg.features["heuristic_tau"],
        embed_dim=cfg.features["heuristic_embed_dim"],
        window_len=cfg.features["heuristic_window_len"],
        kde_bandwidth=cfg.features["heuristic_kde_bandwidth"],
        density_quantile=cfg.features["density_quantile"],
        maxdim=2,
        use_density_filter=True,
    )


def _save_fig(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_point_cloud_views(cfg: ExperimentConfig, out_dir: Path) -> None:
    X, y = _prepare_data(cfg)
    tda_cfg = _robust_tda_cfg(cfg)

    end = min(len(X) - 2, tda_cfg.window_len + 300)
    window_df = X.iloc[end - tda_cfg.window_len : end]
    points = takens_embedding(window_df.to_numpy(), tau=tda_cfg.tau, embed_dim=tda_cfg.embed_dim)
    keep = _kde_keep_mask(points, bandwidth=tda_cfg.kde_bandwidth, quantile=tda_cfg.density_quantile)
    reduced = _pca_2d(points)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(reduced[~keep, 0], reduced[~keep, 1], s=10, alpha=0.35, c="#d62728", label="Filtered out")
    ax.scatter(reduced[keep, 0], reduced[keep, 1], s=12, alpha=0.75, c="#1f77b4", label="Kept")
    ax.set_title("Takens Point Cloud (PCA-2D) with Density Filtering")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best")
    _save_fig(fig, out_dir / "point_clouds" / "point_cloud_density_filter.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(window_df.index, window_df[y.name], lw=1.8, c="#2c7fb8")
    ax.set_title("Target Return Series in Selected Window")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log return")
    _save_fig(fig, out_dir / "point_clouds" / "target_window_timeseries.png")


def plot_feature_distributions(cfg: ExperimentConfig, out_dir: Path) -> None:
    X, y = _prepare_data(cfg)
    tda_cfg = _robust_tda_cfg(cfg)
    feat_df, _ = build_feature_matrix(X, y, tda_cfg, use_univariate_only=False)

    tda_cols = [c for c in feat_df.columns if c.startswith("H") or c.startswith("PI_") or c.startswith("density_")]
    for col in tda_cols:
        fig, ax = plt.subplots(figsize=(7, 4))
        vals = feat_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            continue
        ax.hist(vals, bins=30, color="#3182bd", alpha=0.85, edgecolor="white")
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        _save_fig(fig, out_dir / "features" / f"hist_{col}.png")

    corr = feat_df[tda_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=75, fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title("Correlation Matrix of TDA Features")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_fig(fig, out_dir / "features" / "tda_feature_correlation.png")


def _read_csv(results_dir: Path, name: str) -> pd.DataFrame:
    fp = results_dir / name
    if not fp.exists():
        raise FileNotFoundError(f"Missing required results file: {fp}")
    return pd.read_csv(fp)


def plot_experiment_results(results_dir: Path, out_dir: Path) -> None:
    exp1 = _read_csv(results_dir, "experiment_1_baseline.csv")
    exp2 = _read_csv(results_dir, "experiment_2_multidimensionality.csv")
    exp3 = _read_csv(results_dir, "experiment_3_robustness.csv")
    exp4 = _read_csv(results_dir, "experiment_4_optimization.csv")
    exp5 = _read_csv(results_dir, "experiment_5_shap_importance.csv")

    for metric in ["rmse", "mae", "directional_accuracy"]:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(exp1["model"], exp1[metric], color="#4c78a8")
        ax.set_title(f"Experiment 1: Baseline Comparison ({metric})")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", labelrotation=20)
        _save_fig(fig, out_dir / "experiments" / f"exp1_{metric}.png")

    for metric in ["rmse", "mae", "directional_accuracy"]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(exp2["setting"], exp2[metric], color=["#f58518", "#54a24b"])
        ax.set_title(f"Experiment 2: Impact of Multidimensionality ({metric})")
        ax.set_ylabel(metric)
        _save_fig(fig, out_dir / "experiments" / f"exp2_{metric}.png")

    for metric in ["rmse", "mae", "directional_accuracy"]:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(exp3["phase"], exp3[metric], marker="o", linewidth=2, color="#e45756")
        ax.set_title(f"Experiment 3: Outlier Robustness ({metric})")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", labelrotation=20)
        _save_fig(fig, out_dir / "experiments" / f"exp3_{metric}.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(exp4["method"], exp4["rmse"], color=["#72b7b2", "#b279a2"])
    ax.set_title("Experiment 4: Heuristic vs AOT (RMSE)")
    ax.set_ylabel("rmse")
    _save_fig(fig, out_dir / "experiments" / "exp4_rmse.png")

    if "optimization_seconds" in exp4.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(exp4["method"], exp4["optimization_seconds"], color=["#9d755d", "#bab0ab"])
        ax.set_title("Experiment 4: Optimization Runtime")
        ax.set_ylabel("seconds")
        _save_fig(fig, out_dir / "experiments" / "exp4_optimization_seconds.png")

    top = exp5.nlargest(20, "mean_abs_shap").iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#4c78a8")
    ax.set_title("Experiment 5: Top-20 SHAP Feature Importance")
    ax.set_xlabel("mean |SHAP|")
    _save_fig(fig, out_dir / "experiments" / "exp5_top20_shap.png")


def generate_all_visuals(cfg: ExperimentConfig, visual_dir: str | Path, skip_missing_results: bool = True) -> dict[str, str]:
    out_dir = Path(visual_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(cfg.paths["results_dir"])

    plot_point_cloud_views(cfg, out_dir=out_dir)
    plot_feature_distributions(cfg, out_dir=out_dir)
    exp_status = "generated"
    try:
        plot_experiment_results(results_dir=results_dir, out_dir=out_dir)
    except FileNotFoundError:
        if not skip_missing_results:
            raise
        exp_status = "skipped_missing_results"

    return {
        "visual_dir": str(out_dir.resolve()),
        "results_dir": str(results_dir.resolve()),
        "tda_cfg": str(asdict(_robust_tda_cfg(cfg))),
        "experiment_plots": exp_status,
    }
