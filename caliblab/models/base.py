from abc import ABC, abstractmethod

import torch


class ModelBase(torch.nn.Module, ABC):
    """A base class for models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the model."""
        raise NotImplementedError
