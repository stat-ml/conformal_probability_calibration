from abc import ABC, abstractmethod
from typing import Tuple

from torch.utils.data import DataLoader, Dataset

from .utils import DataSplit


class BaseDataset(ABC):
    """A base class for datasets."""

    train_dataset: Dataset
    test_dataset: Dataset

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the dataset."""
        raise NotImplementedError

    def get_train_loader(
        self, batch_size: int, shuffle: bool = True, num_workers: int = 4
    ) -> DataLoader:
        """Get a DataLoader for the training set."""
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    def get_test_loader(
        self, batch_size: int, shuffle: bool = False, num_workers: int = 4
    ) -> DataLoader:
        """Get a DataLoader for the test set."""
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
        )


def get_sizes(datasplit: DataSplit) -> Tuple[int, int]:
    """Return the test and calibration set sizes for a given DataSplit."""
    test_size = len(datasplit.test_labels)
    cal_size = len(datasplit.cal_labels)
    return test_size, cal_size
