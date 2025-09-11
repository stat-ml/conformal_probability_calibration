import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .base import BaseDataset


class MNISTDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "mnist"

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self._setup()

    def _setup(self):
        self.train_dataset = torchvision.datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transform,
        )
        self.test_dataset = torchvision.datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transform,
        )
