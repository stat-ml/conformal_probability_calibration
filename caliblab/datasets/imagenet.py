from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .base import BaseDataset
from .utils import split_dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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

        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(
                f"ImageNet-mini data not found at {self.data_dir}. "
                "Please download and extract it into 'train' and 'val' subdirectories."
            )

        self.train_dataset = ImageFolder(train_path, transform=self.train_transform)
        original_test_dataset = ImageFolder(val_path, transform=self.test_transform)

        self.cal_dataset, self.test_dataset = split_dataset(
            original_test_dataset, self.cal_ratio, self.seed
        )
