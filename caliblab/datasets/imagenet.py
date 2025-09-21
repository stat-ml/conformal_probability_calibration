from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .base import BaseDataset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImageNetMiniDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "imagenet-mini"

    def __init__(
        self,
        data_dir: str,
        size: int = 224,
    ):
        self.data_dir = Path(data_dir)
        self.size = size
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
        # Downloaded from https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000
        train_path = self.data_dir / "train"
        val_path = self.data_dir / "val"

        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(
                f"ImageNet-mini data not found at {self.data_dir}. "
                "Please download and extract it into 'train' and 'val' subdirectories."
            )

        train_dataset = ImageFolder(train_path, transform=self.test_transform)
        num_classes = len(train_dataset.classes)
        num_train = len(train_dataset)
        total_images = num_train
        print(f"{self.name}: {total_images} images, {num_classes} classes (train={num_train})")
        self.test_dataset = train_dataset
