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
        for i, (inputs, labels) in enumerate(tqdm(loader)):
            inputs = inputs.to(device)
            outputs = self.model(inputs)
            assert outputs.shape[0] == inputs.shape[0]
            all_outputs.append(outputs.cpu().numpy().astype(np.float64))
            all_labels.append(labels.cpu().numpy().astype(np.int64))
            if i >= 10:
                print(outputs.shape, labels.shape)
                break

        return np.concatenate(all_outputs), np.concatenate(all_labels)
