from typing import Any, Dict, Tuple

import torch

from .base import ModelBase

HUB_REPO_ALIASES: Dict[str, str] = {
    "pytorch-cifar-models": "chenyaofo/pytorch-cifar-models",
    "vision": "pytorch/vision",
    "torchvision": "pytorch/vision",
    "repvgg": "DingXiaoH/RepVGG",
}


def _parse_hub_spec(s: str) -> Tuple[str, str]:
    s = s.strip()
    if ":" not in s:
        raise ValueError(
            f"Invalid hub spec '{s}'. Use 'owner/repo:model_fn' or 'alias:model_fn'."
        )
    repo, fn = s.split(":", 1)
    repo = HUB_REPO_ALIASES.get(repo.strip(), repo.strip())
    return repo, fn.strip()


class TorchHubModel(ModelBase):
    """A wrapper for a generic model loaded from a torch.hub spec."""

    def __init__(self, hub_spec: str, **kwargs: Any):
        super().__init__()
        self._name = hub_spec

        repo, fn = _parse_hub_spec(hub_spec)

        if "pretrained" not in kwargs and "weights" not in kwargs:
            kwargs["pretrained"] = True

        self.model = torch.hub.load(repo, fn, **kwargs)

    @property
    def name(self) -> str:
        return self._name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
