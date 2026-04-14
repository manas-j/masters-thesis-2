from __future__ import annotations

import numpy as np
from xgboost import XGBRegressor


def fit_predict_xgb(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, seed: int = 42) -> np.ndarray:
    model = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=seed,
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model.predict(X_val)


def fit_xgb_model(X_train: np.ndarray, y_train: np.ndarray, seed: int = 42) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=seed,
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model
