import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100

from .base import BaseDataset
from .utils import split_dataset


class CIFAR10Dataset(BaseDataset):
    @property
    def name(self) -> str:
        return "cifar10"

    def __init__(self, data_dir: str, cal_ratio: float = 0.3, seed: int = 0):
        self.data_dir = data_dir
        self.cal_ratio = cal_ratio
        self.seed = seed
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        self._setup()

    def _setup(self):
        self.train_dataset = CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transform,
        )
        original_test_dataset = CIFAR10(
            self.data_dir, train=False, download=True, transform=self.transform
        )
        self.cal_dataset, self.test_dataset = split_dataset(
            original_test_dataset, self.cal_ratio, self.seed
        )


class CIFAR100Dataset(BaseDataset):
    @property
    def name(self) -> str:
        return "cifar100"

    def __init__(
        self,
        data_dir: str,
        cal_ratio: float = 0.5,
        seed: int = 0,
        image_size: int = 32,
    ):
        self.data_dir = data_dir
        self.cal_ratio = cal_ratio
        self.seed = seed
        self.image_size = image_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        self._setup()

    def _setup(self):
        self.train_dataset = CIFAR100(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transform,
        )
        original_test_dataset = CIFAR100(
            self.data_dir, train=False, download=True, transform=self.transform
        )

        self.cal_dataset, self.test_dataset = split_dataset(
            original_test_dataset, self.cal_ratio, self.seed
        )
