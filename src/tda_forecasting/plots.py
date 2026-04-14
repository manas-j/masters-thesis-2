from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ripser import ripser

from .config import ExperimentConfig
from .data import load_market_data, standardize_features_train_stats
from .tda_features import TDAFeatureConfig, filtered_embedding_points


def write_quick_experiment_plots(results_dir: str | Path, plots_dir: str | Path) -> None:
    """Regenerate training curves and experiment bar charts from CSV/JSON (no feature recomputation)."""
    root = Path(results_dir)
    dest = Path(plots_dir)
    plot_training_curves(root / "training_curves.json", dest / "training")
    plot_experiment_metrics(root, dest)


def plot_training_curves(curves_json: str | Path, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = Path(curves_json)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    for name, series in data.items():
        train_mse = series.get("train_mse", [])
        val_mse = series.get("val_mse", [])
        if not train_mse and not val_mse:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        epochs = range(1, len(train_mse) + 1)
        if train_mse:
            ax.plot(epochs, train_mse, label="Train MSE (batch mean)")
        if val_mse:
            ax.plot(epochs, val_mse, label="Validation MSE (held-in tail)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.set_title(f"Training curve — {name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        safe = name.replace(" ", "_").replace("/", "-")
        fig.tight_layout()
        fig.savefig(out / f"training_curve_{safe}.png", dpi=150)
        plt.close(fig)


def plot_experiment_metrics(results_dir: str | Path, plots_dir: str | Path) -> None:
    root = Path(results_dir)
    dest = Path(plots_dir) / "experiments"
    dest.mkdir(parents=True, exist_ok=True)

    e1 = root / "experiment_1_baseline.csv"
    if e1.exists():
        df = pd.read_csv(e1)
        if {"model", "rmse", "mae", "directional_accuracy"} <= set(df.columns):
            for metric, fname in (
                ("rmse", "exp1_rmse.png"),
                ("mae", "exp1_mae.png"),
                ("directional_accuracy", "exp1_directional_accuracy.png"),
            ):
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(df["model"].astype(str), df[metric])
                ax.set_ylabel(metric)
                ax.set_title("Experiment 1 — baseline comparison")
                plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
                fig.tight_layout()
                fig.savefig(dest / fname, dpi=150)
                plt.close(fig)

    for i, csv_name in enumerate(
        ("experiment_2_multidimensionality.csv", "experiment_3_robustness.csv"),
        start=2,
    ):
        p = root / csv_name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        metric_cols = [c for c in df.columns if c in ("rmse", "mae", "directional_accuracy")]
        label_col = "setting" if "setting" in df.columns else "phase" if "phase" in df.columns else None
        if not metric_cols or not label_col:
            continue
        for metric in metric_cols:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(df[label_col].astype(str), df[metric])
            ax.set_ylabel(metric)
            ax.set_title(f"Experiment {i} — {metric}")
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
            fig.tight_layout()
            fig.savefig(dest / f"exp{i}_{metric}.png", dpi=150)
            plt.close(fig)

    e4 = root / "experiment_4_optimization.csv"
    if e4.exists():
        df = pd.read_csv(e4)
        if "rmse" in df.columns and "method" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(df["method"].astype(str), df["rmse"])
            ax.set_ylabel("RMSE")
            ax.set_title("Experiment 4 — heuristic vs Optuna")
            fig.tight_layout()
            fig.savefig(dest / "exp4_rmse.png", dpi=150)
            plt.close(fig)
        if "optimization_seconds" in df.columns and "method" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            secs = df["optimization_seconds"].clip(lower=0.0)
            ax.bar(df["method"].astype(str), secs)
            ax.set_ylabel("Optimization time (s)")
            ax.set_title("Experiment 4 — wall-clock (Optuna phase)")
            fig.tight_layout()
            fig.savefig(dest / "exp4_optimization_seconds.png", dpi=150)
            plt.close(fig)

    e5 = root / "experiment_5_shap_importance.csv"
    if e5.exists():
        df = pd.read_csv(e5).head(20)
        if {"feature", "mean_abs_shap"} <= set(df.columns):
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(df["feature"].astype(str)[::-1], df["mean_abs_shap"][::-1])
            ax.set_xlabel("Mean |SHAP|")
            ax.set_title("Experiment 5 — top features (XGBoost surrogate)")
            fig.tight_layout()
            fig.savefig(dest / "exp5_top20_shap.png", dpi=150)
            plt.close(fig)


def plot_tda_feature_histograms(feature_csv: str | Path | None, plots_dir: str | Path) -> None:
    """If a CSV of feature rows is provided, plot histograms; otherwise skip."""
    if feature_csv is None:
        return
    path = Path(feature_csv)
    if not path.exists():
        return
    df = pd.read_csv(path)
    dest = Path(plots_dir) / "features"
    dest.mkdir(parents=True, exist_ok=True)
    num_cols = [c for c in df.columns if c != "target" and df[c].dtype in (np.float64, np.float32, float)]
    for c in num_cols[:24]:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(df[c].dropna(), bins=40, color="steelblue", alpha=0.85)
        ax.set_title(c)
        fig.tight_layout()
        fig.savefig(dest / f"hist_{c}.png", dpi=120)
        plt.close(fig)


def plot_point_cloud_panel(cfg: ExperimentConfig, out_dir: str | Path) -> None:
    dest = Path(out_dir) / "point_clouds"
    dest.mkdir(parents=True, exist_ok=True)
    data_cfg = cfg.data
    bundle = load_market_data(
        start=data_cfg["start"],
        end=data_cfg["end"],
        target_ticker=data_cfg["target_ticker"],
        feature_tickers=data_cfg["feature_tickers"],
    )
    n = len(bundle.X)
    split_idx = max(1, int(n * (1.0 - float(cfg.training["test_size"]))))
    Xs = standardize_features_train_stats(bundle.X, split_idx)
    y = bundle.y
    L = int(cfg.features["heuristic_window_len"])
    tau = int(cfg.features["heuristic_tau"])
    d = int(cfg.features["heuristic_embed_dim"])
    t = len(Xs) - 2
    w = Xs.iloc[t - L + 1 : t + 1]
    cfg_vis = TDAFeatureConfig(
        tau=tau, embed_dim=d, window_len=L, kde_bandwidth=1.0, density_quantile=0.05, maxdim=2, use_density_filter=False
    )
    pts, _ = filtered_embedding_points(w, cfg_vis)
    if pts.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.5, c=np.arange(len(pts)), cmap="viridis")
        ax.set_xlabel("coord 0")
        ax.set_ylabel("coord 1")
        ax.set_title("Joint Takens cloud (first 2 coords, colored by time)")
        fig.tight_layout()
        fig.savefig(dest / "point_cloud_density_filter.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(y.iloc[-400:].to_numpy(), lw=0.8)
    ax.set_title("Target log-return (last 400 obs.)")
    ax.set_xlabel("Time index")
    fig.tight_layout()
    fig.savefig(dest / "target_window_timeseries.png", dpi=150)
    plt.close(fig)


def plot_persistence_diagram_sample(cfg: ExperimentConfig, out_dir: str | Path) -> None:
    dest = Path(out_dir) / "features"
    dest.mkdir(parents=True, exist_ok=True)
    data_cfg = cfg.data
    bundle = load_market_data(
        start=data_cfg["start"],
        end=data_cfg["end"],
        target_ticker=data_cfg["target_ticker"],
        feature_tickers=data_cfg["feature_tickers"],
    )
    n = len(bundle.X)
    split_idx = max(1, int(n * (1.0 - float(cfg.training["test_size"]))))
    Xs = standardize_features_train_stats(bundle.X, split_idx)
    L = int(cfg.features["heuristic_window_len"])
    tda_cfg = TDAFeatureConfig(
        tau=int(cfg.features["heuristic_tau"]),
        embed_dim=int(cfg.features["heuristic_embed_dim"]),
        window_len=L,
        kde_bandwidth=float(cfg.features["heuristic_kde_bandwidth"]),
        density_quantile=float(cfg.features["density_quantile"]),
        maxdim=2,
        use_density_filter=True,
    )
    t = len(Xs) - 2
    w = Xs.iloc[t - L + 1 : t + 1]
    pts, _ = filtered_embedding_points(w, tda_cfg)
    dgms = ripser(pts, maxdim=2)["dgms"]
    fig, ax = plt.subplots(figsize=(5, 5))
    lim = 0.0
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for dim, dgm in enumerate(dgms):
        dgm = dgm[np.isfinite(dgm).all(axis=1)]
        if not dgm.size:
            continue
        ax.scatter(dgm[:, 0], dgm[:, 1], s=12, alpha=0.7, c=colors[dim % len(colors)], label=f"H{dim}")
        lim = max(lim, float(np.nanmax(dgm)))
    lim = max(lim * 1.1, 1e-3)
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title("Sample persistence diagram (density-filtered window)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest / "sample_persistence_diagram.png", dpi=150)
    plt.close(fig)


def plot_feature_correlation_heatmap(feature_csv: str | Path, out_dir: str | Path) -> None:
    path = Path(feature_csv)
    if not path.exists():
        return
    df = pd.read_csv(path)
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    ax.set_title("TDA + lag feature correlation")
    fig.tight_layout()
    dest = Path(out_dir) / "features"
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest / "tda_feature_correlation.png", dpi=150)
    plt.close(fig)


def export_feature_matrix_csv(cfg: ExperimentConfig, out_path: str | Path) -> Path:
    """Materialize multivariate robust-TDA feature table for histogram / correlation plots."""
    data_cfg = cfg.data
    bundle = load_market_data(
        start=data_cfg["start"],
        end=data_cfg["end"],
        target_ticker=data_cfg["target_ticker"],
        feature_tickers=data_cfg["feature_tickers"],
    )
    n = len(bundle.X)
    split_idx = max(1, int(n * (1.0 - float(cfg.training["test_size"]))))
    Xs = standardize_features_train_stats(bundle.X, split_idx)
    tda_cfg = TDAFeatureConfig(
        tau=int(cfg.features["heuristic_tau"]),
        embed_dim=int(cfg.features["heuristic_embed_dim"]),
        window_len=int(cfg.features["heuristic_window_len"]),
        kde_bandwidth=float(cfg.features["heuristic_kde_bandwidth"]),
        density_quantile=float(cfg.features["density_quantile"]),
        maxdim=2,
        use_density_filter=True,
    )
    from .tda_features import build_feature_matrix

    feats, tgt = build_feature_matrix(Xs, bundle.y, tda_cfg, use_univariate_only=False)
    out = feats.copy()
    out["target"] = tgt.values
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    return p
