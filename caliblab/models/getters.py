from .base import ModelBase
from .cifar_hub import CIFARHubModel
from .torch_hub import TorchHubModel
from .torchvision import TorchvisionModel


def get_model(model_name: str, **kwargs) -> ModelBase:
    """
    Factory function to get a model instance by name.

    Args:
        model_name: The name or specifier for the model.
        **kwargs: Additional arguments passed to the model constructor.

    Returns:
        An instance of a ModelBase subclass.
    """
    name = model_name.strip()

    # Case 1: Explicit hub spec: "repo:model"
    if ":" in name:
        return TorchHubModel(name, **kwargs)

    # Case 2: CIFAR hub names: "cifar10_*" / "cifar100_*"
    if name.startswith(("cifar10_", "cifar100_")):
        return CIFARHubModel(name, **kwargs)

    # Case 3: Torchvision factories (default)
    # This will raise an error via torch.hub if the model doesn't exist.
    return TorchvisionModel(name, **kwargs)
