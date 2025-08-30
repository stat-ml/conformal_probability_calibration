from caliblab.datasets.base import BaseDataset
from caliblab.datasets.cifar import CIFAR10Dataset, CIFAR100Dataset
from caliblab.datasets.imagenet import ImageNetMiniDataset
from caliblab.datasets.mnist import MNISTDataset


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
