from typing import Any, Optional, Tuple

import birder
import numpy as np
import torch
from birder.inference.classification import infer_image, infer_batch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from .base import ModelBase
from caliblab.utils.device import get_device


class _BirderModel(ModelBase):
    def __init__(self, model: torch.nn.Module, name: str, model_info: Any):
        super().__init__(model, name)
        self.model_info = model_info
        size = birder.get_size_from_signature(self.model_info.signature)
        self.transform = birder.classification_transform(size, self.model_info.rgb_stats)

    @torch.no_grad()
    def predict(
        self, loader: DataLoader, device: torch.device
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.model.to(device)
        self.model.eval()

        all_outputs = []
        all_labels = []
        for i, (inputs, labels) in enumerate(tqdm(loader)):
            inputs = inputs.to(device)
            (outputs, _) = infer_batch(self.model, inputs, self.transform, return_logits=True)
            assert outputs.shape[0] == inputs.shape[0]
            all_outputs.append(outputs.astype(np.float64))
            all_labels.append(labels.cpu().numpy().astype(np.int64))

        return np.concatenate(all_outputs), np.concatenate(all_labels), None

def BirderModel(
    model_name: str,
    alias: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs: Any,
) -> ModelBase:
    if "inference" not in kwargs:
        kwargs["inference"] = True

    device = get_device()
    net, model_info = birder.load_pretrained_model(model_name, device=device.type, **kwargs)

    return _BirderModel(net, alias or model_name, model_info)
