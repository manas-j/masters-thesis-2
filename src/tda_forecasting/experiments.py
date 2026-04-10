from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
import time

import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig
from .data import load_market_data
from .metrics import directional_accuracy, mae, rmse
from .models.lstm import fit_predict_lstm
from .models.tree import fit_predict_xgb, fit_xgb_model
from .tda_features import TDAFeatureConfig, build_feature_matrix


@dataclass(slots=True)
class EvalResult:
    model: str
    rmse: float
    mae: float
    directional_accuracy: float


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _to_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len : i])
        ys.append(y[i])
    return np.asarray(Xs), np.asarray(ys)


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
    return bundle.X, bundle.y


def _walk_forward_splits(n_obs: int, n_splits: int, min_train_size: int, test_size: int) -> list[tuple[slice, slice]]:
    splits: list[tuple[slice, slice]] = []
    for i in range(n_splits):
        train_end = min_train_size + i * test_size
        test_end = train_end + test_size
        if test_end > n_obs:
            break
        splits.append((slice(0, train_end), slice(train_end, test_end)))
    return splits


def _run_baseline_suite(features: pd.DataFrame, target: pd.Series, training_cfg: dict) -> list[EvalResult]:
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=training_cfg["test_size"], shuffle=False)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    seq_len = training_cfg["lstm_seq_len"]
    X_train_seq, y_train_seq = _to_sequences(X_train_sc, y_train.to_numpy(), seq_len)
    X_test_seq, y_test_seq = _to_sequences(X_test_sc, y_test.to_numpy(), seq_len)

    yhat_lstm = fit_predict_lstm(
        X_train_seq,
        y_train_seq,
        X_test_seq,
        epochs=training_cfg["lstm_epochs"],
        batch_size=training_cfg["batch_size"],
        hidden_size=training_cfg["lstm_hidden_size"],
    )
    yhat_xgb = fit_predict_xgb(X_train_sc, y_train.to_numpy(), X_test_sc)

    return [
        _evaluate(y_test_seq, yhat_lstm, "LSTM"),
        _evaluate(y_test.to_numpy(), yhat_xgb, "XGBoost"),
    ]


def _run_lstm_walk_forward(features: pd.DataFrame, target: pd.Series, training_cfg: dict, model_name: str) -> EvalResult:
    n = len(features)
    holdout = max(32, int(np.ceil(training_cfg["test_size"] * n)))
    min_train = max(256, n - 3 * holdout)
    splits = _walk_forward_splits(n_obs=n, n_splits=3, min_train_size=min_train, test_size=holdout)
    if not splits:
        return _run_baseline_suite(features, target, training_cfg)[0]

    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    for tr, te in splits:
        X_train, X_test = features.iloc[tr], features.iloc[te]
        y_train, y_test = target.iloc[tr], target.iloc[te]
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        seq_len = training_cfg["lstm_seq_len"]
        X_train_seq, y_train_seq = _to_sequences(X_train_sc, y_train.to_numpy(), seq_len)
        X_test_seq, y_test_seq = _to_sequences(X_test_sc, y_test.to_numpy(), seq_len)
        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            continue
        pred = fit_predict_lstm(
            X_train_seq,
            y_train_seq,
            X_test_seq,
            epochs=training_cfg["lstm_epochs"],
            batch_size=training_cfg["batch_size"],
            hidden_size=training_cfg["lstm_hidden_size"],
        )
        y_true_all.append(y_test_seq)
        y_pred_all.append(pred)

    if not y_true_all:
        return _run_baseline_suite(features, target, training_cfg)[0]

    y_true_cat = np.concatenate(y_true_all)
    y_pred_cat = np.concatenate(y_pred_all)
    return _evaluate(y_true_cat, y_pred_cat, model_name)


def _optimize_tda_params(X: pd.DataFrame, y: pd.Series, cfg: ExperimentConfig, use_density: bool) -> dict[str, float]:
    train_cfg = cfg.training
    opt_cfg = cfg.optimization

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
        feats, target = build_feature_matrix(X, y, tda_cfg, use_univariate_only=False)
        X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=train_cfg["test_size"], shuffle=False)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        seq_len = train_cfg["lstm_seq_len"]
        X_train_seq, y_train_seq = _to_sequences(X_train_sc, y_train.to_numpy(), seq_len)
        X_test_seq, y_test_seq = _to_sequences(X_test_sc, y_test.to_numpy(), seq_len)
        pred = fit_predict_lstm(
            X_train_seq,
            y_train_seq,
            X_test_seq,
            epochs=train_cfg["optuna_lstm_epochs"],
            batch_size=train_cfg["batch_size"],
            hidden_size=train_cfg["lstm_hidden_size"],
        )
        return rmse(y_test_seq, pred)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.training["seed"]))
    study.optimize(objective, n_trials=cfg.optimization["n_trials"], show_progress_bar=False)
    return study.best_params


def run_all_experiments(cfg: ExperimentConfig) -> dict[str, pd.DataFrame]:
    _seed_everything(cfg.training["seed"])
    X, y = _prepare_data(cfg)
    out_dir = Path(cfg.paths["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    heuristic = TDAFeatureConfig(
        tau=cfg.features["heuristic_tau"],
        embed_dim=cfg.features["heuristic_embed_dim"],
        window_len=cfg.features["heuristic_window_len"],
        kde_bandwidth=cfg.features["heuristic_kde_bandwidth"],
        density_quantile=cfg.features["density_quantile"],
        maxdim=2,
        use_density_filter=False,
    )

    robust = TDAFeatureConfig(**{**asdict(heuristic), "use_density_filter": True})

    # Exp 1: Baseline comparison
    lag_only = pd.DataFrame({
        "lag_1": y.shift(1),
        "lag_5": y.rolling(5).mean(),
        "lag_20": y.rolling(20).mean(),
    }).dropna()
    y_lag = y.loc[lag_only.index]
    exp1 = pd.DataFrame([asdict(r) for r in _run_baseline_suite(lag_only, y_lag, cfg.training)])

    tda_plain_X, tda_plain_y = build_feature_matrix(X, y, heuristic, use_univariate_only=False)
    tda_rob_X, tda_rob_y = build_feature_matrix(X, y, robust, use_univariate_only=False)

    plain_lstm = _run_lstm_walk_forward(tda_plain_X, tda_plain_y, cfg.training, "TDA-LSTM")
    rob_lstm = _run_lstm_walk_forward(tda_rob_X, tda_rob_y, cfg.training, "Robust Density-Filtered TDA-LSTM")

    exp1 = pd.concat(
        [
            exp1,
            pd.DataFrame([asdict(plain_lstm)]),
            pd.DataFrame([asdict(rob_lstm)]),
        ],
        ignore_index=True,
    )

    # Exp 2: Impact of multidimensionality
    uni_X, uni_y = build_feature_matrix(X, y, robust, use_univariate_only=True)
    mul_X, mul_y = build_feature_matrix(X, y, robust, use_univariate_only=False)
    exp2 = pd.DataFrame(
        [
            asdict(_run_baseline_suite(uni_X, uni_y, cfg.training)[0]) | {"setting": "1D target only"},
            asdict(_run_baseline_suite(mul_X, mul_y, cfg.training)[0]) | {"setting": "Joint multivariate"},
        ]
    )

    # Exp 3: Outlier robustness ablation + stress test
    noisy_X = _noise_injection(X, p=cfg.experiments["stress_noise_prob"], scale=cfg.experiments["stress_noise_scale"])
    base_clean = _run_lstm_walk_forward(tda_plain_X, tda_plain_y, cfg.training, "TDA-LSTM")
    robust_clean = _run_lstm_walk_forward(tda_rob_X, tda_rob_y, cfg.training, "Robust Density-Filtered TDA-LSTM")
    noisy_plain_X, noisy_plain_y = build_feature_matrix(noisy_X, y.loc[noisy_X.index], heuristic, use_univariate_only=False)
    noisy_rob_X, noisy_rob_y = build_feature_matrix(noisy_X, y.loc[noisy_X.index], robust, use_univariate_only=False)
    base_noisy = _run_lstm_walk_forward(noisy_plain_X, noisy_plain_y, cfg.training, "TDA-LSTM")
    robust_noisy = _run_lstm_walk_forward(noisy_rob_X, noisy_rob_y, cfg.training, "Robust Density-Filtered TDA-LSTM")

    exp3 = pd.DataFrame(
        [
            asdict(base_clean) | {"phase": "A_clean_no_filter"},
            asdict(robust_clean) | {"phase": "B_clean_density_filter"},
            asdict(base_noisy) | {"phase": "C_noisy_no_filter"},
            asdict(robust_noisy) | {"phase": "C_noisy_density_filter"},
        ]
    )

    # Exp 4: Heuristic vs AOT (Optuna)
    t0 = time.perf_counter()
    best = _optimize_tda_params(X, y, cfg, use_density=True)
    opt_seconds = time.perf_counter() - t0
    opt_tda = TDAFeatureConfig(maxdim=2, use_density_filter=True, **best)
    opt_X, opt_y = build_feature_matrix(X, y, opt_tda, use_univariate_only=False)
    heur_result = _run_lstm_walk_forward(tda_rob_X, tda_rob_y, cfg.training, "Robust Density-Filtered TDA-LSTM")
    opt_result = _run_lstm_walk_forward(opt_X, opt_y, cfg.training, "Robust Density-Filtered TDA-LSTM")
    exp4 = pd.DataFrame(
        [
            asdict(heur_result)
            | {
                "method": "Heuristic params",
                "tau": heuristic.tau,
                "embed_dim": heuristic.embed_dim,
                "window_len": heuristic.window_len,
                "kde_bandwidth": heuristic.kde_bandwidth,
                "optimization_seconds": 0.0,
            },
            asdict(opt_result) | {"method": "Optuna params", "optimization_seconds": opt_seconds, **best},
        ]
    )

    # Exp 5: SHAP interpretability (XGBoost surrogate)
    x_train, x_test, y_train, y_test = train_test_split(tda_rob_X, tda_rob_y, test_size=cfg.training["test_size"], shuffle=False)
    surrogate = fit_xgb_model(x_train.to_numpy(), y_train.to_numpy(), seed=cfg.training["seed"])
    explainer = shap.TreeExplainer(surrogate)
    shap_values = explainer.shap_values(x_test.to_numpy())
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

    return outputs
