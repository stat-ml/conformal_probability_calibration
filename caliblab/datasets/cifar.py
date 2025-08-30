import torch
from torch.utils.data import DataLoader, random_split

from .base import BaseDataset

import torchvision
import torchvision.transforms as transforms


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
        self.train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transform,
        )
        original_test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transform,
        )

        test_size = len(original_test_dataset)
        cal_size = int(test_size * self.cal_ratio)
        test_size = test_size - cal_size
        self.cal_dataset, self.test_dataset = random_split(
            original_test_dataset,
            [cal_size, test_size],
            generator=torch.Generator().manual_seed(self.seed),
        )

    def get_train_loader(
        self, batch_size: int, shuffle: bool = True, num_workers: int = 4
    ) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def get_cal_loader(
        self, batch_size: int, shuffle: bool = False, num_workers: int = 4
    ) -> DataLoader:
        return DataLoader(
            self.cal_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def get_test_loader(
        self, batch_size: int, shuffle: bool = False, num_workers: int = 4
    ) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )


class CIFAR100Dataset(BaseDataset):
    @property
    def name(self) -> str:
        return "cifar100"

    def __init__(self, data_dir: str, cal_ratio: float = 0.3, seed: int = 0):
        self.data_dir = data_dir
        self.cal_ratio = cal_ratio
        self.seed = seed
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        self._setup()

    def _setup(self):
        self.train_dataset = torchvision.datasets.CIFAR100(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transform,
        )
        original_test_dataset = torchvision.datasets.CIFAR100(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transform,
        )

        test_size = len(original_test_dataset)
        cal_size = int(test_size * self.cal_ratio)
        test_size = test_size - cal_size
        self.cal_dataset, self.test_dataset = random_split(
            original_test_dataset,
            [cal_size, test_size],
            generator=torch.Generator().manual_seed(self.seed),
        )

    def get_train_loader(
        self, batch_size: int, shuffle: bool = True, num_workers: int = 4
    ) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def get_cal_loader(
        self, batch_size: int, shuffle: bool = False, num_workers: int = 4
    ) -> DataLoader:
        return DataLoader(
            self.cal_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def get_test_loader(
        self, batch_size: int, shuffle: bool = False, num_workers: int = 4
    ) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
