from abc import ABC, abstractmethod

from torch.utils.data import DataLoader


class BaseDataset(ABC):
    """A base class for datasets."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the dataset."""
        raise NotImplementedError

    @abstractmethod
    def get_train_loader(
        self, batch_size: int, shuffle: bool = True, num_workers: int = 0
    ) -> DataLoader:
        """Get a DataLoader for the training set.
        Args:
            batch_size (int): The batch size to use.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
            num_workers (int, optional): The number of workers to use for data loading.
        Returns:
            DataLoader: A DataLoader for the training set.
        """
        raise NotImplementedError

    @abstractmethod
    def get_cal_loader(
        self, batch_size: int, shuffle: bool = False, num_workers: int = 0
    ) -> DataLoader:
        """Get a DataLoader for the calibration set.
        Args:
            batch_size (int): The batch size to use.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
            num_workers (int, optional): The number of workers to use for data loading.
        Returns:
            DataLoader: A DataLoader for the calibration set.
        """
        raise NotImplementedError

    @abstractmethod
    def get_test_loader(
        self, batch_size: int, shuffle: bool = False, num_workers: int = 0
    ) -> DataLoader:
        """Get a DataLoader for the test set.
        Args:
            batch_size (int): The batch size to use.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
            num_workers (int, optional): The number of workers to use for data loading.
        Returns:
            DataLoader: A DataLoader for the test set.
        """
        raise NotImplementedError
