from typing import Any

import torch

from .base import ModelBase


class TorchvisionModel(ModelBase):
    """A wrapper for torchvision models loaded from torch.hub."""

    def __init__(self, model_name: str, **kwargs: Any):
        super().__init__()
        self._name = model_name

        # Default to pretrained weights if user didn't specify
        if "pretrained" not in kwargs and "weights" not in kwargs:
            kwargs["pretrained"] = True

        self.model = torch.hub.load("pytorch/vision", model_name, **kwargs)

    @property
    def name(self) -> str:
        return self._name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
