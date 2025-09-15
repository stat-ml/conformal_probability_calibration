from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import ViTForImageClassification

from .base import ModelBase


class _ViTLogitsWrapper(nn.Module):
    def __init__(self, vit_model: ViTForImageClassification):
        super().__init__()
        self.vit_model = vit_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit_model(x).logits


def ViTModel(
    model_name: str,
    alias: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs: Any,
) -> ModelBase:
    model = ViTForImageClassification.from_pretrained(
        model_name, cache_dir=cache_dir, **kwargs
    )
    wrapper = _ViTLogitsWrapper(model)
    name = alias or model_name
    return ModelBase(wrapper, name)
