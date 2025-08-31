from typing import Any, Optional

import torch

from .base import ModelBase


class HubModel(ModelBase):
    """A general wrapper for a model from Torch Hub."""

    def __init__(
        self,
        repo_or_dir: str,
        model_name: str,
        alias: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self._repo_or_dir = repo_or_dir
        self._model_name = model_name
        self._alias = alias

        if cache_dir:
            torch.hub.set_dir(cache_dir)

        # Default to pretrained weights if not specified by the user
        if "pretrained" not in kwargs and "weights" not in kwargs:
            kwargs["pretrained"] = True

        self.model = torch.hub.load(repo_or_dir, model_name, **kwargs)

    @property
    def name(self) -> str:
        # For hub models, the alias is more important as the model_name
        # might not be descriptive enough on its own.
        return self._alias or self._model_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
