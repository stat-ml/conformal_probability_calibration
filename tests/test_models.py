import pytest
import torch

from caliblab.models import get_model
from caliblab.models.cifar_hub import CIFARHubModel
from caliblab.models.torch_hub import TorchHubModel
from caliblab.models.torchvision import TorchvisionModel

# Helper to skip tests if no internet connection is available for torch.hub
try:
    torch.hub.load("pytorch/vision", "resnet18", pretrained=False)
    has_hub_connection = True
except Exception:
    has_hub_connection = False

skip_if_no_hub = pytest.mark.skipif(
    not has_hub_connection, reason="torch.hub requires internet connection"
)


@skip_if_no_hub
@pytest.mark.parametrize(
    "model_name, expected_class",
    [
        ("resnet18", TorchvisionModel),
        ("cifar10_resnet20", CIFARHubModel),
        ("pytorch/vision:resnet18", TorchHubModel),
    ],
)
def test_get_model_factory(model_name, expected_class):
    """Test that the get_model factory returns the correct model class."""
    model = get_model(model_name, pretrained=False)
    assert isinstance(model, expected_class)


@skip_if_no_hub
def test_model_forward_pass():
    """Test that a loaded model can perform a forward pass."""
    model = get_model("resnet18", pretrained=False)
    # The model expects 3-channel images
    dummy_input = torch.randn(2, 3, 32, 32)
    output = model(dummy_input)
    # ResNet18 has 1000 output classes by default
    assert output.shape == (2, 1000)
