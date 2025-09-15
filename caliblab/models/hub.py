from typing import Any, Optional

import torch

from .base import ModelBase


def HubModel(
    repo_or_dir: str,
    model_name: str,
    alias: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs: Any,
) -> ModelBase:
    if cache_dir:
        torch.hub.set_dir(cache_dir)

    # Default to pretrained weights if not specified by the user
    if "pretrained" not in kwargs and "weights" not in kwargs:
        kwargs["pretrained"] = True

    model = torch.hub.load(repo_or_dir, model_name, **kwargs)
    name = alias or model_name
    return ModelBase(model, name)
