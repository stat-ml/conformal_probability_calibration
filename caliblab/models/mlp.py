import math
import os

import torch
import torch.nn as nn
from .base import ModelBase



# ========= Model: MLP with optional random Fourier features =========
class FourierFeatures(nn.Module):
    """
    Map R^2 -> R^{2*B + 2} via random Fourier features (RFF).
    Helps with locality and many-class separation on grids.
    """

    def __init__(self, in_dim: int = 2, n_frequencies: int = 64, scale: float = 1.0):
        super().__init__()
        self.B = nn.Parameter(
            torch.randn(in_dim, n_frequencies) * scale, requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,2), B: (2,B) -> (N,B)
        proj = x @ self.B  # (N,B)
        return torch.cat(
            [x, torch.sin(2 * math.pi * proj), torch.cos(2 * math.pi * proj)], dim=-1
        )


class MLPClassifier(ModelBase):
    def __init__(
        self, n_classes: int, hidden: int = 256, n_rff: int = 64, rff_scale: float = 0.5
    ):
        super().__init__(model=None, name=f"mlp_{n_classes}")
        self.ff = (
            FourierFeatures(2, n_frequencies=n_rff, scale=rff_scale)
            if n_rff > 0
            else nn.Identity()
        )
        in_dim = 2 if isinstance(self.ff, nn.Identity) else (2 + 2 * n_rff)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, n_classes),
        )
        stacked_model = nn.ModuleList(
            [self.ff, self.net]
        )
        self.model = stacked_model

    def forward(self, x):
        x = self.ff(x)
        return self.net(x)

    def save_weights(self, path: str):
        """Save only state_dict(). Recreate model with SAME hyperparams before loading."""
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str, map_location="cpu", strict: bool = True):
        """Load only state_dict(). Model must have the SAME hyperparams."""
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict, strict=strict)
