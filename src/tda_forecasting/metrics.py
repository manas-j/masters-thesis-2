from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    signs_match = np.sign(y_true) == np.sign(y_pred)
    return float(np.mean(signs_match))
