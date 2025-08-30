from typing import Any, Optional

import torch
from transformers import ViTForImageClassification

from .base import ModelBase


class ViTModel(ModelBase):
    """
    A wrapper for a Hugging Face ViT model for image classification.
    """

    def __init__(
        self,
        model_name: str,
        alias: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self._model_name = model_name
        self._alias = alias
        self.model = ViTForImageClassification.from_pretrained(
            model_name, cache_dir=cache_dir, **kwargs
        )

    @property
    def name(self) -> str:
        return self._alias or self._model_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass and returns the logits.
        The output from ViTForImageClassification is a dictionary-like object.
        """
        outputs = self.model(x)
        return outputs.logits
