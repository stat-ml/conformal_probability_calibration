from typing import Any

import torch

from .base import ModelBase


class CIFARHubModel(ModelBase):
    """A wrapper for models from the pytorch-cifar-models hub."""

    def __init__(self, model_name: str, **kwargs: Any):
        super().__init__()
        self._name = model_name

        if not model_name.startswith(("cifar10_", "cifar100_")):
            raise ValueError(
                "CIFARHubModel names must start with 'cifar10_' or 'cifar100_'"
            )

        if "pretrained" not in kwargs and "weights" not in kwargs:
            kwargs["pretrained"] = True

        self.model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", model_name, **kwargs
        )

    @property
    def name(self) -> str:
        return self._name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
