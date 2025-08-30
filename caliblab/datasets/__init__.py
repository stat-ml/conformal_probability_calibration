from .base import BaseDataset
from .cifar import CIFAR10Dataset, CIFAR100Dataset
from .imagenet import ImageNetMiniDataset
from .mnist import MNISTDataset


def dataset_getter(name: str, **kwargs) -> BaseDataset:
    """
    Returns a dataset instance by name.
    """
    name = name.lower()
    if name == "cifar10":
        return CIFAR10Dataset(**kwargs)
    elif name == "cifar100":
        return CIFAR100Dataset(**kwargs)
    elif name == "mnist":
        return MNISTDataset(**kwargs)
    elif name == "imagenet-mini":
        return ImageNetMiniDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")


__all__ = [
    "BaseDataset",
    "CIFAR10Dataset",
    "CIFAR100Dataset",
    "MNISTDataset",
    "ImageNetMiniDataset",
    "dataset_getter",
]
