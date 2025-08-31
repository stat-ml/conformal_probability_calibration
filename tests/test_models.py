import pytest
import torch

from caliblab.models import get_model
from caliblab.models.hub import HubModel
from caliblab.models.vit import ViTModel

# Helper to skip tests if no internet connection is available for torch.hub
skip_if_no_hub = pytest.mark.skipif(
    not hasattr(torch.hub, "check_integrity"),
    reason="torch.hub requires internet connection",
)


@skip_if_no_hub
@pytest.mark.parametrize(
    "source, name, repo, expected_class",
    [
        ("torch_hub", "resnet18", "pytorch/vision", HubModel),
        (
            "torch_hub",
            "cifar10_resnet20",
            "chenyaofo/pytorch-cifar-models",
            HubModel,
        ),
        ("vit", "google/vit-base-patch16-224", None, ViTModel),
    ],
)
def test_get_model_factory(source, name, repo, expected_class):
    """Test that the get_model factory returns the correct model class."""
    model = get_model(name=name, source=source, repo=repo, pretrained=False)
    assert isinstance(model, expected_class)


@skip_if_no_hub
def test_model_alias():
    """Test that the model's name property uses the alias if provided."""
    # Test without alias
    model_no_alias = get_model(
        name="resnet18", source="torch_hub", repo="pytorch/vision", pretrained=False
    )
    assert model_no_alias.name == "resnet18"

    # Test with alias
    model_with_alias = get_model(
        name="resnet18",
        source="torch_hub",
        repo="pytorch/vision",
        alias="MyResNet18",
        pretrained=False,
    )
    assert model_with_alias.name == "MyResNet18"


@skip_if_no_hub
def test_model_forward_pass():
    """Test that a loaded model can perform a forward pass."""
    model = get_model(
        name="resnet18", source="torch_hub", repo="pytorch/vision", pretrained=False
    )
    # The model expects 3-channel images
    dummy_input = torch.randn(2, 3, 32, 32)
    output = model(dummy_input)
    # ResNet18 has 1000 output classes by default
    assert output.shape == (2, 1000)
