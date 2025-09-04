from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base import CalibratorBase


class TemperatureScaling(CalibratorBase):
    def __init__(self, temperature: float = 1.0, lr: float = 0.01, max_iter: int = 50):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.lr = lr
        self.max_iter = max_iter

    @property
    def name(self) -> str:
        return "temp_scaling"

    def fit(
        self,
        *,
        logits: Optional[np.ndarray] = None,
        probs: Optional[np.ndarray] = None,
        y_true: np.ndarray,
        **kwargs,
    ) -> "TemperatureScaling":
        if logits is None:
            if probs is None:
                raise ValueError("Either logits or probs must be provided.")
            # Convert probs to logits
            logits = np.log(probs + 1e-12)

        logits_tensor = torch.from_numpy(logits)
        labels_tensor = torch.from_numpy(y_true)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=self.lr, max_iter=self.max_iter)

        def eval():
            optimizer.zero_grad()
            loss = criterion(logits_tensor / self.temperature, labels_tensor)
            loss.backward()
            return loss

        optimizer.step(eval)
        self._mark_fitted()
        return self

    def predict_proba(
        self,
        *,
        logits: Optional[np.ndarray] = None,
        probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self.check_fitted()
        if logits is None:
            if probs is None:
                raise ValueError("Either logits or probs must be provided.")
            logits = np.log(probs + 1e-12)

        with torch.no_grad():
            logits_tensor = torch.from_numpy(logits)
            scaled_logits = logits_tensor / self.temperature
            return torch.softmax(scaled_logits, dim=1).numpy()
