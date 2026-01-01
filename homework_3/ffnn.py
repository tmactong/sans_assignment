from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from dataset import load_features_and_labels
import settings
from optimal_hyper_params import non_shuffle_features, non_shuffle_features_lasso


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    train_size: float = 0.7
    batch_size: int = 128
    max_epochs: int = 500
    lr: float = 1e-3
    hidden_sizes: Tuple[int, ...] = (64, 32)


class FFNNRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_sizes: Sequence[int] = (64, 32)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _to_numpy_1d(a: np.ndarray | pd.Series | pd.DataFrame) -> np.ndarray:
    if isinstance(a, pd.DataFrame):
        a = a.iloc[:, 0]
    if isinstance(a, pd.Series):
        a = a.to_numpy()
    a = np.asarray(a)
    return a.reshape(-1)


def _to_numpy_2d(a: np.ndarray | pd.DataFrame) -> np.ndarray:
    if isinstance(a, pd.DataFrame):
        return a.to_numpy()
    return np.asarray(a)


def fit_predict_ffnn(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    cfg: TrainConfig,
    selected_features: Optional[Sequence[str]] = None,
) -> dict:
    """Train FFNN and return metrics + predictions aligned by original index."""

    if selected_features is not None:
        X = X.loc[:, list(selected_features)]

    # Split (for non-shuffled time split, set shuffle=False)
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=cfg.train_size,
        shuffle=False
    )

    train_idx = x_train.index
    test_idx = x_test.index

    X_train_np = _to_numpy_2d(x_train)
    X_test_np = _to_numpy_2d(x_test)
    y_train_np = _to_numpy_1d(y_train)
    y_test_np = _to_numpy_1d(y_test)

    # Scale features (fit on train only)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_np)
    X_test_s = scaler.transform(X_test_np)

    # Torch tensors
    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_np, dtype=torch.float32).view(-1, 1)

    train_loader_train = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    # IMPORTANT: keep order fixed when computing metrics / overall alignment
    train_loader_eval = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=max(cfg.batch_size, 256),
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=max(cfg.batch_size, 256),
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FFNNRegressor(
        in_dim=X_train_t.shape[1],
        hidden_sizes=cfg.hidden_sizes,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    def run_epoch(loader: DataLoader, train: bool) -> Tuple[float, np.ndarray, np.ndarray]:
        model.train(train)
        total = 0.0
        ys: list[np.ndarray] = []
        ps: list[np.ndarray] = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)

            pred = model(xb)
            loss = criterion(pred, yb)

            if train:
                loss.backward()
                optimizer.step()

            total += loss.item() * xb.size(0)
            ys.append(yb.detach().cpu().numpy())
            ps.append(pred.detach().cpu().numpy())

        y_all = np.vstack(ys).reshape(-1)
        p_all = np.vstack(ps).reshape(-1)
        return total / len(loader.dataset), y_all, p_all

    for _epoch in range(1, cfg.max_epochs + 1):
        run_epoch(train_loader_train, train=True)

    # Final predictions
    _, y_tr, p_tr = run_epoch(train_loader_eval, train=False)
    _, y_te, p_te = run_epoch(test_loader, train=False)

    train_r2 = float(r2_score(y_tr, p_tr))
    test_r2 = float(r2_score(y_te, p_te))
    train_rmse = root_mean_squared_error(y_tr, p_tr)
    test_rmse = root_mean_squared_error(y_te, p_te)

    # Overall metrics on concatenated predictions, aligned by original index
    pred_index = np.concatenate([train_idx.to_numpy(), test_idx.to_numpy()])
    pred_values = np.concatenate([p_tr, p_te])
    y_index = X.index

    y_series = pd.Series(_to_numpy_1d(y), index=y_index).sort_index()
    pred_series = pd.Series(pred_values, index=pred_index).sort_index()

    # Ensure identical ordering
    pred_series = pred_series.reindex(y_series.index)

    overall_r2 = float(r2_score(y_series.to_numpy(), pred_series.to_numpy()))
    overall_rmse = float(root_mean_squared_error(y_series.to_numpy(), pred_series.to_numpy()))

    return {
        "train_r2": round(train_r2, 3),
        "train_rmse": round(train_rmse, 3),
        "test_r2": round(test_r2, 3),
        "test_rmse": round(test_rmse, 3),
        "overall_r2": round(overall_r2, 3),
        "overall_rmse": round(overall_rmse, 3),
        "n_features": X.shape[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)

    # FFNN hyperparameters (keep simple; you can grid-search outside if needed)
    parser.add_argument("--hidden", type=str, default="64,32", help="Comma-separated hidden layer sizes")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=500)

    args = parser.parse_args()

    set_seed(args.seed)

    hidden_sizes = tuple(int(x) for x in args.hidden.split(",") if x.strip())

    cfg = TrainConfig(
        train_size=args.train_size,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lr=args.lr,
        hidden_sizes=hidden_sizes,
    )

    # Load your dataset (expects: X as DataFrame, y as Series/array)
    X, y = load_features_and_labels(settings.DatasetFile)


    # Train on full features
    metrics = fit_predict_ffnn(X, y, cfg, selected_features=non_shuffle_features_lasso[args.train_size])
    # metrics = fit_predict_ffnn(X, y, cfg)
    print(f"FFNN (non-shuffled), train_size={cfg.train_size}")
    print("|Model|train r2|train rmse|test r2|test rmse|overall r2|overall rmse|")
    print("|---|---|---|---|---|---|---|")
    print(
        f"|{args.train_size}|{metrics['train_r2']}|{metrics['train_rmse']}|{metrics['test_r2']}|{metrics['test_rmse']}|"
        f"{metrics['overall_r2']}|{metrics['overall_rmse']}|"
    )


if __name__ == "__main__":
    # Ensure deterministic-ish behavior (still some GPU nondeterminism possible)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()