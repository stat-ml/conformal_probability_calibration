from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import ImageFolder

from .base import BaseDataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class _TransformedDataset(Dataset):
    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


class ImageNetMiniDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "imagenet-mini"

    def __init__(
        self,
        data_dir: str,
        cal_ratio: float = 0.5,
        size: int = 224,
        seed: int = 0,
    ):
        self.data_dir = Path(data_dir)
        self.cal_ratio = cal_ratio
        self.size = size
        self.seed = seed
        self.train_transform = T.Compose(
            [
                T.RandomResizedCrop(size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self.test_transform = T.Compose(
            [
                T.Resize(size + 32),
                T.CenterCrop(size),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self._setup()

    def _setup(self):
        # Assumes a standard ImageNet folder structure with 'train' and 'val' folders.
        train_path = self.data_dir / "train"
        val_path = self.data_dir / "val"

        self.train_dataset = ImageFolder(
            train_path, transform=self.train_transform
        )
        original_test_dataset = ImageFolder(val_path)

        num_test = len(original_test_dataset)
        cal_size = int(self.cal_ratio * num_test)
        test_size = num_test - cal_size

        cal_subset, test_subset = random_split(
            original_test_dataset,
            [cal_size, test_size],
            generator=torch.Generator().manual_seed(self.seed),
        )

        self.cal_dataset = _TransformedDataset(
            cal_subset, transform=self.test_transform
        )
        self.test_dataset = _TransformedDataset(
            test_subset, transform=self.test_transform
        )

    def get_train_loader(
        self, batch_size: int, shuffle: bool = True, num_workers: int = 4
    ) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    def get_cal_loader(
        self, batch_size: int, shuffle: bool = False, num_workers: int = 4
    ) -> DataLoader:
        return DataLoader(
            self.cal_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    def get_test_loader(
        self, batch_size: int, shuffle: bool = False, num_workers: int = 4
    ) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
