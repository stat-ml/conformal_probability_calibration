import numpy as np
import pytest
from ..utils.computations import softmax
from tqdm import tqdm

from caliblab.datasets import dataset_getter
from caliblab.metrics.classification import Accuracy
from caliblab.models import get_model

# Helper to skip tests if no internet connection is available for torch.hub
try:
    torch.hub.load("pytorch/vision", "resnet18", pretrained=False)
    has_hub_connection = True
except Exception:
    has_hub_connection = False

skip_if_no_hub = pytest.mark.skipif(
    not has_hub_connection, reason="torch.hub requires internet connection"
)


def _get_predictions(model, loader, device):
    """Helper function to get model predictions for a data loader."""
    model.to(device)
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = softmax(outputs)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


@skip_if_no_hub
@pytest.mark.parametrize(
    "dataset_name, model_repo, model_name, min_accuracy",
    [
        (
            "cifar10",
            "chenyaofo/pytorch-cifar-models",
            "cifar10_resnet20",
            0.90,
        ),
        (
            "cifar100",
            "chenyaofo/pytorch-cifar-models",
            "cifar100_resnet20",
            0.67,
        ),
    ],
)
def test_model_benchmark_accuracy(
    dataset_name, model_repo, model_name, min_accuracy, tmp_path
):
    """
    Tests that a pretrained model achieves a minimum accuracy on a benchmark dataset.
    """
    # 1. Load the dataset
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    try:
        dataset = dataset_getter(dataset_name, data_dir=str(data_dir))
    except Exception as e:
        pytest.skip(f"Skipping {dataset_name} due to download or setup error: {e}")

    # 2. Load the pretrained model
    model = get_model(
        name=model_name, source="torch_hub", repo=model_repo, pretrained=True
    )

    # 3. Create a data loader
    test_loader = dataset.get_test_loader(batch_size=256, num_workers=4)

    # 4. Get predictions
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    probs, labels = _get_predictions(model, test_loader, device)

    # 5. Calculate accuracy using the new metric
    accuracy_metric = Accuracy()
    accuracy = accuracy_metric(probs=probs, y_true=labels)
    print(f"Accuracy of {model_name} on {dataset_name}: {accuracy:.4f}")

    # 6. Assert that accuracy is above the threshold
    assert accuracy > min_accuracy, (
        f"Expected accuracy for {model_name} on {dataset_name} to be > {min_accuracy}, but got {accuracy:.4f}"
    )
