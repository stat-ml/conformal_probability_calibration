import torchvision.transforms as transforms
from torchvision.datasets import INaturalist

from .base import BaseDataset


class INaturalistDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "inaturalist"

    def __init__(
        self,
        data_dir: str,
        image_size: int = 224,
        version: str = "2021_train_mini",
    ):
        self.data_dir = data_dir
        self.image_size = image_size
        self.version = version
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self._setup()

    def _setup(self):
        self.train_dataset = INaturalist(
            self.data_dir,
            version=self.version,
            download=True,
            transform=self.transform,
        )
        self.test_dataset = INaturalist(
            self.data_dir,
            version="2021_valid",
            download=True,
            transform=self.transform,
        )
