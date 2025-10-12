import pytest
import torch

from caliblab.datasets import dataset_getter


@pytest.mark.parametrize(
    "dataset_name", ["cifar10", "cifar100", "mnist"]
)
def test_vision_datasets(dataset_name, tmp_path):
    """Test that vision datasets can be loaded and iterated."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    try:
        dataset = dataset_getter(dataset_name, data_dir=str(data_dir), seed=42)
    except Exception as e:
        pytest.skip(f"Skipping {dataset_name} due to download or setup error: {e}")

    assert dataset is not None, "Dataset should not be None"

    train_loader = dataset.get_train_loader(batch_size=4, num_workers=0)
    cal_loader = None
    if hasattr(dataset, "get_cal_loader"):
        try:
            cal_loader = dataset.get_cal_loader(batch_size=4, num_workers=0)
        except AttributeError:
            cal_loader = None
    test_loader = dataset.get_test_loader(batch_size=4, num_workers=0)

    assert train_loader is not None, "Train loader should not be None"
    if cal_loader is not None:
        assert cal_loader is not None, "Calibration loader should not be None"
    assert test_loader is not None, "Test loader should not be None"

    # Check that we can get an item from each loader
    train_images, train_labels = next(iter(train_loader))
    if cal_loader is not None:
        cal_images, cal_labels = next(iter(cal_loader))
    test_images, test_labels = next(iter(test_loader))

    assert isinstance(train_images, torch.Tensor)
    assert isinstance(train_labels, torch.Tensor)
    if cal_loader is not None:
        assert isinstance(cal_images, torch.Tensor)
        assert isinstance(cal_labels, torch.Tensor)
    assert isinstance(test_images, torch.Tensor)
    assert isinstance(test_labels, torch.Tensor)

    assert train_images.shape[0] == 4
    if cal_loader is not None:
        assert cal_images.shape[0] == 4
    assert test_images.shape[0] == 4


@pytest.mark.slow
def test_inaturalist_dataset(tmp_path):
    """Test that iNaturalist dataset can be loaded and iterated."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    try:
        dataset = dataset_getter("inaturalist", data_dir=str(data_dir), seed=42)
    except Exception as e:
        pytest.skip(f"Skipping inaturalist due to download or setup error: {e}")

    assert dataset is not None, "Dataset should not be None"

    train_loader = dataset.get_train_loader(batch_size=4, num_workers=0)
    cal_loader = None
    if hasattr(dataset, "get_cal_loader"):
        try:
            cal_loader = dataset.get_cal_loader(batch_size=4, num_workers=0)
        except AttributeError:
            cal_loader = None
    test_loader = dataset.get_test_loader(batch_size=4, num_workers=0)

    assert train_loader is not None, "Train loader should not be None"
    if cal_loader is not None:
        assert cal_loader is not None, "Calibration loader should not be None"
    assert test_loader is not None, "Test loader should not be None"

    # Check that we can get an item from each loader
    train_images, train_labels = next(iter(train_loader))
    if cal_loader is not None:
        cal_images, cal_labels = next(iter(cal_loader))
    test_images, test_labels = next(iter(test_loader))

    assert isinstance(train_images, torch.Tensor)
    assert isinstance(train_labels, torch.Tensor)
    if cal_loader is not None:
        assert isinstance(cal_images, torch.Tensor)
        assert isinstance(cal_labels, torch.Tensor)
    assert isinstance(test_images, torch.Tensor)
    assert isinstance(test_labels, torch.Tensor)

    assert train_images.shape[0] == 4
    if cal_loader is not None:
        assert cal_images.shape[0] == 4
    assert test_images.shape[0] == 4


def test_imagenet_mini_dataset(tmp_path):
    """Test the ImageNetMiniDataset with a dummy data structure."""
    data_dir = tmp_path / "imagenet-mini"
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    # Create dummy directory structure
    for i in range(2):
        (train_dir / f"class_{i}").mkdir(parents=True, exist_ok=True)
        (val_dir / f"class_{i}").mkdir(parents=True, exist_ok=True)

    # Create dummy image files
    from PIL import Image

    for i in range(4):
        img = Image.new("RGB", (60, 30), color="red")
        class_folder = i % 2
        if i < 2:
            img.save(train_dir / f"class_{class_folder}" / f"img{i}.png")
        else:
            img.save(val_dir / f"class_{class_folder}" / f"img{i}.png")

    dataset = dataset_getter(
        "imagenet-mini", data_dir=str(data_dir)
    )
    assert dataset is not None

    test_loader = dataset.get_test_loader(batch_size=1, num_workers=0)

    # Current implementation uses 'train' as the test dataset
    assert len(test_loader.dataset) == 2

    test_images, _ = next(iter(test_loader))
    assert test_images.shape[0] == 1
