from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def fit_predict_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_size: int = 64,
    seed: int = 42,
    val_frac: float = 0.12,
    return_history: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, list[float]]]:
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = len(X_train)
    n_va = max(int(val_frac * n), min(64, n // 8))
    n_va = min(n_va, n - 32)
    if n_va < 1 or n - n_va < 8:
        X_tr, y_tr = X_train, y_train
        X_va_i, y_va_i = X_train, y_train
    else:
        X_tr, y_tr = X_train[:-n_va], y_train[:-n_va]
        X_va_i, y_va_i = X_train[-n_va:], y_train[-n_va:]

    model = LSTMRegressor(input_size=X_train.shape[-1], hidden_size=hidden_size).to(device)
    pin = device == "cuda"
    ds_tr = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, pin_memory=pin)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    X_va_t = torch.tensor(X_va_i, dtype=torch.float32).to(device)
    y_va_t = torch.tensor(y_va_i, dtype=torch.float32).to(device)

    history: dict[str, list[float]] = {"train_mse": [], "val_mse": []}

    epoch_iter = tqdm(range(epochs), desc="LSTM", leave=False, unit="ep")
    for _ in epoch_iter:
        model.train()
        epoch_losses: list[float] = []
        for xb, yb in loader_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            va_pred = model(X_va_t)
            history["train_mse"].append(float(np.mean(epoch_losses)) if epoch_losses else float("nan"))
            history["val_mse"].append(float(loss_fn(va_pred, y_va_t).cpu()))

    model.eval()
    with torch.no_grad():
        x_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_hat = model(x_val).cpu().numpy()

    if return_history:
        return y_hat, history
    return y_hat
