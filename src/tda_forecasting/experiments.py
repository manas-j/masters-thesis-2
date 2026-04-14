from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig
from .data import load_market_data, standardize_features_train_stats
from .metrics import directional_accuracy, mae, rmse
from .models.lstm import fit_predict_lstm
from .models.tree import fit_predict_xgb, fit_xgb_model
from .tda_features import TDAFeatureConfig, build_feature_matrix

H0_FEATURE_COLS = ("H0_entropy", "H0_amplitude", "H0_npoints")


@dataclass(slots=True)
class EvalResult:
    model: str
    rmse: float
    mae: float
    directional_accuracy: float


def _to_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(X)
    if n <= seq_len:
        return (
            np.empty((0, seq_len, X.shape[-1]), dtype=np.float64),
            np.empty((0,), dtype=y.dtype),
        )
    # Windows X[i-seq_len:i] for i = seq_len..n-1 (length n - seq_len).
    starts = np.arange(0, n - seq_len, dtype=np.intp)[:, None]
    offs = np.arange(seq_len, dtype=np.intp)[None, :]
    idx = starts + offs
    Xs = np.asarray(X[idx], dtype=np.float64, order="C")
    return Xs, y[seq_len:]


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> EvalResult:
    return EvalResult(
        model=model_name,
        rmse=rmse(y_true, y_pred),
        mae=mae(y_true, y_pred),
        directional_accuracy=directional_accuracy(y_true, y_pred),
    )


def _noise_injection(df: pd.DataFrame, p: float = 0.02, scale: float = 8.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    k = max(1, int(p * n))
    idx = rng.choice(n, size=k, replace=False)
    for c in out.columns:
        sigma = out[c].std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            continue
        spikes = rng.normal(loc=0.0, scale=scale * sigma, size=k)
        out.iloc[idx, out.columns.get_loc(c)] += spikes
    return out


def _prepare_data(cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.Series]:
    data_cfg = cfg.data
    bundle = load_market_data(
        start=data_cfg["start"],
        end=data_cfg["end"],
        target_ticker=data_cfg["target_ticker"],
        feature_tickers=data_cfg["feature_tickers"],
    )
    n = len(bundle.X)
    split_idx = max(1, int(n * (1.0 - float(cfg.training["test_size"]))))
    X_scaled = standardize_features_train_stats(bundle.X, split_idx)
    return X_scaled, bundle.y


def _run_baseline_suite(
    features: pd.DataFrame,
    target: pd.Series,
    training_cfg: dict,
    seed: int,
    lstm_label: str = "LSTM",
    collect_lstm_history: bool = False,
) -> tuple[list[EvalResult], dict[str, list[float]] | None]:
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=training_cfg["test_size"], shuffle=False
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    seq_len = int(training_cfg["lstm_seq_len"])
    X_train_seq, y_train_seq = _to_sequences(X_train_sc, y_train.to_numpy(), seq_len)

    X_train_xgb = X_train_sc[seq_len:]
    y_train_xgb = y_train.to_numpy()[seq_len:]

    X_test_xgb = X_test_sc[seq_len:]
    y_test_xgb = y_test.to_numpy()[seq_len:]

    X_test_seq, y_test_seq = _to_sequences(X_test_sc, y_test.to_numpy(), seq_len)

    hist_out: dict[str, list[float]] | None = None
    if collect_lstm_history:
        yhat_lstm, hist_out = fit_predict_lstm(
            X_train_seq,
            y_train_seq,
            X_test_seq,
            epochs=int(training_cfg["lstm_epochs"]),
            batch_size=int(training_cfg["batch_size"]),
            hidden_size=int(training_cfg["lstm_hidden_size"]),
            seed=seed,
            return_history=True,
        )
    else:
        yhat_lstm = fit_predict_lstm(
            X_train_seq,
            y_train_seq,
            X_test_seq,
            epochs=int(training_cfg["lstm_epochs"]),
            batch_size=int(training_cfg["batch_size"]),
            hidden_size=int(training_cfg["lstm_hidden_size"]),
            seed=seed,
            return_history=False,
        )

    yhat_xgb = fit_predict_xgb(X_train_xgb, y_train_xgb, X_test_xgb, seed=seed)

    return [
        _evaluate(y_test_seq, yhat_lstm, lstm_label),
        _evaluate(y_test_xgb, yhat_xgb, "XGBoost"),
    ], hist_out


def _h0_train_activity(X_train: pd.DataFrame) -> float:
    cols = [c for c in H0_FEATURE_COLS if c in X_train.columns]
    if not cols:
        return 0.0
    return float(np.mean(X_train[cols].std(axis=0, ddof=0)))


def _tda_trial_scores(
    feats: pd.DataFrame,
    target: pd.Series,
    train_cfg: dict,
    opt_cfg: dict,
    seed: int,
    lstm_epochs: int,
) -> tuple[float, float, pd.DataFrame]:
    """
    Returns (composite_loss, rmse_only, X_train_unscaled) for one TDA+LSTM fit.
    composite penalizes flat H0 columns on the training fold so search explores
    regions where H0 summaries vary across time (better captured by the model).
    """
    if len(feats) < 64:
        return 1e6, 1e6, feats.iloc[:0]

    X_train, X_test, y_train, y_test = train_test_split(
        feats, target, test_size=train_cfg["test_size"], shuffle=False
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    seq_len = int(train_cfg["lstm_seq_len"])
    if len(X_train) <= seq_len or len(X_test) <= seq_len:
        return 1e6, 1e6, X_train

    X_train_seq, y_train_seq = _to_sequences(X_train_sc, y_train.to_numpy(), seq_len)
    X_test_seq, y_test_seq = _to_sequences(X_test_sc, y_test.to_numpy(), seq_len)
    pred = fit_predict_lstm(
        X_train_seq,
        y_train_seq,
        X_test_seq,
        epochs=int(lstm_epochs),
        batch_size=int(train_cfg["batch_size"]),
        hidden_size=int(train_cfg["lstm_hidden_size"]),
        seed=int(seed),
        return_history=False,
    )
    rmse_val = rmse(y_test_seq, pred)
    w = float(opt_cfg.get("h0_activity_weight", 0.06))
    act = _h0_train_activity(X_train)
    composite = float(rmse_val + w * np.exp(-act))
    return composite, rmse_val, X_train


def _grid_search_pools(opt_cfg: dict, gs: dict) -> dict[str, list]:
    """Build Cartesian factor pools; YAML lists override Optuna min/max ranges."""
    pools: dict[str, list] = {}
    if "tau" in gs and isinstance(gs["tau"], list):
        pools["tau"] = [int(x) for x in gs["tau"]]
    else:
        pools["tau"] = list(range(int(opt_cfg["tau_min"]), int(opt_cfg["tau_max"]) + 1))

    if "embed_dim" in gs and isinstance(gs["embed_dim"], list):
        pools["embed_dim"] = [int(x) for x in gs["embed_dim"]]
    else:
        pools["embed_dim"] = list(range(int(opt_cfg["embed_dim_min"]), int(opt_cfg["embed_dim_max"]) + 1))

    if "window_len" in gs and isinstance(gs["window_len"], list):
        pools["window_len"] = [int(x) for x in gs["window_len"]]
    else:
        pools["window_len"] = list(
            range(int(opt_cfg["window_min"]), int(opt_cfg["window_max"]) + 1, max(1, (int(opt_cfg["window_max"]) - int(opt_cfg["window_min"])) // 8 or 1))
        )
        if len(pools["window_len"]) > 12:
            step = max(1, len(pools["window_len"]) // 12)
            pools["window_len"] = pools["window_len"][::step][:12]

    if "kde_bandwidth" in gs and isinstance(gs["kde_bandwidth"], list):
        pools["kde_bandwidth"] = [float(x) for x in gs["kde_bandwidth"]]
    else:
        lo, hi = float(opt_cfg["kde_bw_min"]), float(opt_cfg["kde_bw_max"])
        pools["kde_bandwidth"] = np.linspace(lo, hi, num=6, dtype=np.float64).round(4).tolist()

    if "density_quantile" in gs and isinstance(gs["density_quantile"], list):
        pools["density_quantile"] = [float(x) for x in gs["density_quantile"]]
    else:
        pools["density_quantile"] = [0.02, 0.04, 0.06, 0.10, 0.14, 0.18]

    return pools


def _sample_grid_candidates(
    pools: dict[str, list],
    max_configs: int,
    rng: np.random.Generator,
) -> list[dict[str, float | int]]:
    keys = ("tau", "embed_dim", "window_len", "kde_bandwidth", "density_quantile")
    full = list(itertools.product(*[pools[k] for k in keys]))
    if len(full) <= max_configs:
        return [dict(zip(keys, combo)) for combo in full]
    idx = rng.choice(len(full), size=max_configs, replace=False)
    return [dict(zip(keys, full[i])) for i in idx]


def _optimize_tda_params(X: pd.DataFrame, y: pd.Series, cfg: ExperimentConfig, use_density: bool) -> dict[str, float]:
    train_cfg = cfg.training
    opt_cfg = cfg.optimization
    seed = int(cfg.training["seed"])
    rng = np.random.default_rng(seed)

    warmstart: list[dict[str, float | int]] = []
    gs = opt_cfg.get("grid_search") or {}
    if isinstance(gs, dict) and gs.get("enabled", False):
        pools = _grid_search_pools(opt_cfg, gs)
        max_cf = int(gs.get("max_configs", 64))
        grid_epochs = int(gs.get("lstm_epochs", train_cfg.get("grid_search_lstm_epochs", max(4, int(train_cfg["optuna_lstm_epochs"]) // 2))))
        candidates = _sample_grid_candidates(pools, max_cf, rng)
        scored: list[tuple[float, float, dict[str, float | int]]] = []
        for params in tqdm(candidates, desc="TDA grid search", leave=False, unit="cfg"):
            tda_cfg = TDAFeatureConfig(
                tau=int(params["tau"]),
                embed_dim=int(params["embed_dim"]),
                window_len=int(params["window_len"]),
                kde_bandwidth=float(params["kde_bandwidth"]),
                density_quantile=float(params["density_quantile"]),
                maxdim=2,
                use_density_filter=use_density,
            )
            try:
                feats, targ = build_feature_matrix(X, y, tda_cfg, use_univariate_only=False)
            except ValueError:
                continue
            comp, raw, _ = _tda_trial_scores(feats, targ, train_cfg, opt_cfg, seed, grid_epochs)
            scored.append((comp, raw, params))
        scored.sort(key=lambda t: t[0])
        top_k = int(gs.get("top_k_for_warmstart", 6))
        warmstart = [p for _, _, p in scored[:top_k]]

    def objective(trial: optuna.Trial) -> float:
        tda_cfg = TDAFeatureConfig(
            tau=trial.suggest_int("tau", opt_cfg["tau_min"], opt_cfg["tau_max"]),
            embed_dim=trial.suggest_int("embed_dim", opt_cfg["embed_dim_min"], opt_cfg["embed_dim_max"]),
            window_len=trial.suggest_int("window_len", opt_cfg["window_min"], opt_cfg["window_max"]),
            kde_bandwidth=trial.suggest_float("kde_bandwidth", opt_cfg["kde_bw_min"], opt_cfg["kde_bw_max"]),
            density_quantile=trial.suggest_float("density_quantile", 0.01, 0.2),
            maxdim=2,
            use_density_filter=use_density,
        )
        try:
            feats, targ = build_feature_matrix(X, y, tda_cfg, use_univariate_only=False)
        except ValueError:
            return 1e6
        comp, _, _ = _tda_trial_scores(
            feats,
            targ,
            train_cfg,
            opt_cfg,
            seed,
            int(train_cfg["optuna_lstm_epochs"]),
        )
        return comp

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    for params in warmstart:
        study.enqueue_trial(
            {
                "tau": int(params["tau"]),
                "embed_dim": int(params["embed_dim"]),
                "window_len": int(params["window_len"]),
                "kde_bandwidth": float(params["kde_bandwidth"]),
                "density_quantile": float(params["density_quantile"]),
            }
        )
    study.optimize(objective, n_trials=int(cfg.optimization["n_trials"]), show_progress_bar=True)
    return study.best_params


def run_all_experiments(cfg: ExperimentConfig) -> dict[str, pd.DataFrame]:
    stage = tqdm(total=6, desc="Pipeline", unit="stage")
    try:
        return _run_all_experiments_impl(cfg, stage)
    finally:
        stage.close()


def _run_all_experiments_impl(cfg: ExperimentConfig, stage: tqdm) -> dict[str, pd.DataFrame]:
    X, y = _prepare_data(cfg)
    out_dir = Path(cfg.paths["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.training["seed"])
    stage.set_postfix_str("data loaded")
    stage.update(1)

    heuristic = TDAFeatureConfig(
        tau=int(cfg.features["heuristic_tau"]),
        embed_dim=int(cfg.features["heuristic_embed_dim"]),
        window_len=int(cfg.features["heuristic_window_len"]),
        kde_bandwidth=float(cfg.features["heuristic_kde_bandwidth"]),
        density_quantile=float(cfg.features["density_quantile"]),
        maxdim=2,
        use_density_filter=False,
    )

    robust = TDAFeatureConfig(**{**asdict(heuristic), "use_density_filter": True})

    training_curves: dict[str, dict[str, list[float]]] = {}

    # Exp 1: Baseline comparison
    lag_only = pd.DataFrame(
        {
            "lag_1": y.shift(1),
            "lag_5": y.rolling(5).mean(),
            "lag_20": y.rolling(20).mean(),
        }
    ).dropna()
    y_lag = y.loc[lag_only.index]
    res_lag, h_lag = _run_baseline_suite(
        lag_only, y_lag, cfg.training, seed, lstm_label="LSTM", collect_lstm_history=True
    )
    if h_lag:
        training_curves["LSTM_lag_only"] = h_lag

    tda_plain_X, tda_plain_y = build_feature_matrix(X, y, heuristic, use_univariate_only=False)
    tda_rob_X, tda_rob_y = build_feature_matrix(X, y, robust, use_univariate_only=False)

    plain_res, h_plain = _run_baseline_suite(
        tda_plain_X, tda_plain_y, cfg.training, seed, lstm_label="TDA-LSTM", collect_lstm_history=True
    )
    rob_res, h_rob = _run_baseline_suite(
        tda_rob_X, tda_rob_y, cfg.training, seed, lstm_label="Robust Density-Filtered TDA-LSTM", collect_lstm_history=True
    )
    if h_plain:
        training_curves["TDA-LSTM"] = h_plain
    if h_rob:
        training_curves["Robust_TDA-LSTM"] = h_rob

    exp1 = pd.DataFrame([asdict(r) for r in res_lag])
    exp1 = pd.concat(
        [
            exp1,
            pd.DataFrame([asdict(plain_res[0])]),
            pd.DataFrame([asdict(rob_res[0])]),
        ],
        ignore_index=True,
    )
    stage.set_postfix_str("exp1 baseline")
    stage.update(1)

    # Exp 2: Impact of multidimensionality (multivariate row matches robust TDA matrix above).
    uni_X, uni_y = build_feature_matrix(X, y, robust, use_univariate_only=True)
    mul_X, mul_y = tda_rob_X, tda_rob_y
    exp2 = pd.DataFrame(
        [
            asdict(_run_baseline_suite(uni_X, uni_y, cfg.training, seed, collect_lstm_history=False)[0][0])
            | {"setting": "1D target only"},
            asdict(_run_baseline_suite(mul_X, mul_y, cfg.training, seed, collect_lstm_history=False)[0][0])
            | {"setting": "Joint multivariate"},
        ]
    )
    stage.set_postfix_str("exp2 multidim")
    stage.update(1)

    # Exp 3: Outlier robustness ablation + stress test
    noisy_X = _noise_injection(X, p=float(cfg.experiments["stress_noise_prob"]), scale=float(cfg.experiments["stress_noise_scale"]), seed=seed)
    base_clean = plain_res[0]
    robust_clean = rob_res[0]
    noisy_plain_X, noisy_plain_y = build_feature_matrix(noisy_X, y.loc[noisy_X.index], heuristic, use_univariate_only=False)
    noisy_rob_X, noisy_rob_y = build_feature_matrix(noisy_X, y.loc[noisy_X.index], robust, use_univariate_only=False)
    base_noisy = _run_baseline_suite(noisy_plain_X, noisy_plain_y, cfg.training, seed, collect_lstm_history=False)[0][0]
    robust_noisy = _run_baseline_suite(noisy_rob_X, noisy_rob_y, cfg.training, seed, collect_lstm_history=False)[0][0]

    exp3 = pd.DataFrame(
        [
            asdict(base_clean) | {"phase": "A_clean_no_filter"},
            asdict(robust_clean) | {"phase": "B_clean_density_filter"},
            asdict(base_noisy) | {"phase": "C_noisy_no_filter"},
            asdict(robust_noisy) | {"phase": "C_noisy_density_filter"},
        ]
    )
    stage.set_postfix_str("exp3 robustness")
    stage.update(1)

    # Exp 4: Heuristic vs Optuna
    t_opt0 = time.perf_counter()
    best = _optimize_tda_params(X, y, cfg, use_density=True)
    opt_seconds = time.perf_counter() - t_opt0
    opt_tda = TDAFeatureConfig(maxdim=2, use_density_filter=True, **best)
    opt_X, opt_y = build_feature_matrix(X, y, opt_tda, use_univariate_only=False)
    heur_result = _run_baseline_suite(tda_rob_X, tda_rob_y, cfg.training, seed, collect_lstm_history=False)[0][0]
    opt_result = _run_baseline_suite(opt_X, opt_y, cfg.training, seed, collect_lstm_history=False)[0][0]
    exp4 = pd.DataFrame(
        [
            asdict(heur_result)
            | {
                "method": "Heuristic params",
                "tau": heuristic.tau,
                "embed_dim": heuristic.embed_dim,
                "window_len": heuristic.window_len,
                "optimization_seconds": 0.0,
            },
            asdict(opt_result)
            | {"method": "Optuna params", "optimization_seconds": opt_seconds, **best},
        ]
    )
    stage.set_postfix_str("exp4 optimization")
    stage.update(1)

    # Exp 5: SHAP interpretability (XGBoost surrogate)
    x_train, x_test, y_train, y_test = train_test_split(
        tda_rob_X, tda_rob_y, test_size=cfg.training["test_size"], shuffle=False
    )
    surrogate = fit_xgb_model(x_train.to_numpy(), y_train.to_numpy(), seed=seed)
    explainer = shap.TreeExplainer(surrogate)
    shap_raw = explainer.shap_values(x_test.to_numpy())
    shap_values = np.asarray(shap_raw)
    if shap_values.ndim == 3:
        shap_values = shap_values.mean(axis=0)
    imp = np.mean(np.abs(shap_values), axis=0)
    exp5 = pd.DataFrame({"feature": x_test.columns, "mean_abs_shap": imp}).sort_values("mean_abs_shap", ascending=False)

    outputs = {
        "experiment_1_baseline.csv": exp1,
        "experiment_2_multidimensionality.csv": exp2,
        "experiment_3_robustness.csv": exp3,
        "experiment_4_optimization.csv": exp4,
        "experiment_5_shap_importance.csv": exp5,
    }

    for name, df in outputs.items():
        df.to_csv(out_dir / name, index=False)

    curves_path = out_dir / "training_curves.json"
    with curves_path.open("w", encoding="utf-8") as f:
        json.dump(training_curves, f, indent=2)

    stage.set_postfix_str("exp5 + saved")
    stage.update(1)
    return outputs


def load_experiment_tables(results_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(results_dir)
    out: dict[str, pd.DataFrame] = {}
    for name in (
        "experiment_1_baseline.csv",
        "experiment_2_multidimensionality.csv",
        "experiment_3_robustness.csv",
        "experiment_4_optimization.csv",
        "experiment_5_shap_importance.csv",
    ):
        p = root / name
        if p.exists():
            out[name] = pd.read_csv(p)
    return out
