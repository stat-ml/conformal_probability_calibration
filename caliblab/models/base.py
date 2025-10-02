from abc import ABC
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class ModelBase(ABC, nn.Module):
    def __init__(self, model: nn.Module, name: str):
        super().__init__()
        self.model = model
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @torch.no_grad()
    def predict(
        self, loader: DataLoader, device: torch.device
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.model.to(device)
        self.model.eval()

        all_outputs = []
        all_labels = []
        all_probs = []
        for i, out in enumerate(tqdm(loader)):
            inputs = out[0]
            if len(out) == 3:
                labels = out[1]
                probs = out[2]
            else:
                labels = out[1]
                probs = None
            inputs = inputs.to(device)
            outputs = self.forward(inputs)
            assert outputs.shape[0] == inputs.shape[0]
            all_outputs.append(outputs.cpu().numpy().astype(np.float64))
            all_labels.append(labels.cpu().numpy().astype(np.int64))
            if probs is not None:
                all_probs.append(probs.cpu().numpy().astype(np.float64))


        return np.concatenate(all_outputs), np.concatenate(all_labels), None if len(all_probs) == 0 else np.concatenate(all_probs)
