from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.studio.datasets import get_dataset, load_dataframe


def run_nn_regression(dataset_id: str, *, epochs: int = 40) -> dict[str, Any]:
    """PyTorch MLP regression baseline for numeric targets."""
    import torch
    from torch import nn

    meta = get_dataset(dataset_id)
    if meta["task"] != "regression":
        return {"error": "nn_regression requires a regression target", "dataset_id": dataset_id}

    df = load_dataframe(dataset_id)
    target = meta["target_column"]
    X = df.drop(columns=[target]).select_dtypes(include=[np.number]).fillna(0)
    y = pd.to_numeric(df[target], errors="coerce").fillna(0).to_numpy(dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X.to_numpy(), y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy().ravel()
    mae = float(np.mean(np.abs(pred - y_test)))
    r2 = float(1 - np.sum((pred - y_test) ** 2) / np.sum((y_test - y_test.mean()) ** 2))

    return {
        "dataset_id": dataset_id,
        "plugin": "nn_regression",
        "val_mae": mae,
        "val_r2": r2,
        "epochs": epochs,
    }
