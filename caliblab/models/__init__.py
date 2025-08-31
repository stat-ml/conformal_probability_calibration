from typing import Optional

from .base import ModelBase
from .hub import HubModel
from .vit import ViTModel


def get_model(
    name: str,
    source: str,
    alias: Optional[str] = None,
    cache_dir: Optional[str] = None,
    repo: Optional[str] = None,
    **kwargs,
) -> ModelBase:
    """
    Factory function to get a model instance by name and source.
    """
    source = source.lower().strip()
    if source == "vit":
        return ViTModel(name, alias=alias, cache_dir=cache_dir, **kwargs)
    elif source == "torch_hub":
        if repo is None:
            raise ValueError(
                "For 'torch_hub' source, the 'repo' parameter is required."
            )
        return HubModel(repo, name, alias=alias, cache_dir=cache_dir, **kwargs)
    else:
        raise ValueError(f"Unknown model source: '{source}'")


__all__ = ["ModelBase", "get_model", "HubModel", "ViTModel"]
