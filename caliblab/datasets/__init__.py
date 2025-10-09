from .base import BaseDataset
from .cifar import CIFAR10Dataset, CIFAR100Dataset
from .imagenet import ImageNetMiniDataset
from .inaturalist import INaturalistDataset
from .mnist import MNISTDataset
from .synthetic import Synthetic2DClassifier


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
    elif name == "inaturalist":
        return INaturalistDataset(**kwargs)
    elif name == "synthetic":
        return Synthetic2DClassifier(kwargs["n_classes"])
    else:
        raise ValueError(f"Unknown dataset: {name}")


__all__ = [
    "BaseDataset",
    "CIFAR10Dataset",
    "CIFAR100Dataset",
    "MNISTDataset",
    "ImageNetMiniDataset",
    "INaturalistDataset",
    "dataset_getter",
    "Synthetic2DClassifier"
]
