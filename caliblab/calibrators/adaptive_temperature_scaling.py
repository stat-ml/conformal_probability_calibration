from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..utils.computations import softmax
from .base import CalibratorBase


class _FeatureToTemperature(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...] = (128, 64)):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure temperature is strictly positive
        return self.softplus(self.net(x)) + 1e-6


class AdaptiveTemperatureScaling(CalibratorBase):
    """
    Adaptive temperature scaling that predicts a per-sample temperature from features.

    If no features are provided, uses logits themselves as features.
    Trains end-to-end to minimize NLL on the calibration split.
    """

    def __init__(
        self,
        *,
        hidden_dims: Tuple[int, ...] = (128, 64),
        lr: float = 5e-4,
        weight_decay: float = 0.0,
        max_epochs: int = 100,
        batch_size: int = 256,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model: Optional[_FeatureToTemperature] = None

    @property
    def name(self) -> str:
        return "ada_temp_scaling"

    def fit(
        self,
        *,
        logits: Optional[np.ndarray] = None,
        y_true: np.ndarray,
        features: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "AdaptiveTemperatureScaling":
        if logits is None:
            raise ValueError("Logits must be provided.")

        x_np = features if features is not None else logits
        if x_np.ndim != 2:
            raise ValueError("Features/logits must be a 2D array [N, D].")

        n_samples, n_classes = logits.shape
        input_dim = x_np.shape[1]

        x = torch.from_numpy(x_np.astype(np.float32))
        y = torch.from_numpy(y_true.astype(np.int64))
        l = torch.from_numpy(logits.astype(np.float32))

        self.model = _FeatureToTemperature(input_dim=input_dim, hidden_dims=self.hidden_dims)
        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        dataset = torch.utils.data.TensorDataset(x, l, y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        self.model.train()
        for _ in range(self.max_epochs):
            for xb, lb, yb in loader:
                xb = xb.to(self.device)
                lb = lb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                t = self.model(xb)  # [B,1]
                scaled_logits = lb / t
                loss = criterion(scaled_logits, yb)
                loss.backward()
                optimizer.step()

        self._mark_fitted()
        return self

    def predict_proba(
        self,
        *,
        logits: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self.check_fitted()
        if self.model is None:
            raise RuntimeError("Model not initialized.")
        if logits is None:
            raise ValueError("Logits must be provided.")

        x_np = features if features is not None else logits
        x = torch.from_numpy(x_np.astype(np.float32)).to(self.device)
        l = torch.from_numpy(logits.astype(np.float32)).to(self.device)

        self.model.eval()
        with torch.no_grad():
            t = self.model(x)
            scaled = (l / t).cpu().numpy()
        return softmax(scaled)


__all__ = ["AdaptiveTemperatureScaling"]


