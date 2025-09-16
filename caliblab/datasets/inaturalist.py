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
        image_size: int = 336
    ):
        self.data_dir = data_dir
        self.image_size = image_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )
        self._setup()

    def _setup(self):
        self.test_dataset = INaturalist(
            self.data_dir,
            version="2021_valid",
            download=True,
            transform=self.transform,
        )
