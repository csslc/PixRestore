"""DINOv2 loading and intermediate feature extraction."""

from __future__ import annotations

import types
from pathlib import Path

import torch
from torch.nn import functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DINOV2_MODELS = {
    "dinov2s": "dinov2_vits14",
    "dinov2b": "dinov2_vitb14",
    "dinov2l": "dinov2_vitl14",
    "dinov2g": "dinov2_vitg14",
}
DINOV2_HUB = "facebookresearch/dinov2"


def _optional_path(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_dinov2(
    encoder_type: str,
    device: torch.device,
    *,
    repository: str | None = None,
    checkpoint: str | None = None,
) -> torch.nn.Module:
    """Load DINOv2 from torch.hub, or from a user-provided local clone / checkpoint."""
    try:
        model_name = DINOV2_MODELS[encoder_type]
    except KeyError as error:
        raise ValueError(f"Unsupported encoder: {encoder_type}") from error

    repository = _optional_path(repository)
    checkpoint = _optional_path(checkpoint)
    model = torch.hub.load(
        repository or DINOV2_HUB,
        model_name,
        source="local" if repository else "github",
        trust_repo=True,
        pretrained=checkpoint is None,
    )
    if checkpoint:
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"DINOv2 checkpoint not found: {path}")
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        model.load_state_dict(state, strict=True)

    def forward_with_features(self, image, masks=None):
        features = {}
        tokens = self.prepare_tokens_with_masks(image, masks)
        for index, block in enumerate(self.blocks):
            tokens = block(tokens)
            features[index] = tokens[:, 1:]
        return features, self.norm(tokens)[:, 1:]

    model.forward_with_features = types.MethodType(forward_with_features, model)
    model.requires_grad_(False)
    return model.to(device).eval()


def preprocess_dinov2(image: torch.Tensor, size: int) -> torch.Tensor:
    """Convert an image in [-1, 1] to normalized DINOv2 input."""
    image = F.interpolate(
        image.add(1).mul(0.5),
        size=(size, size),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp_(0, 1)
    mean = image.new_tensor(IMAGENET_MEAN)[None, :, None, None]
    std = image.new_tensor(IMAGENET_STD)[None, :, None, None]
    return (image - mean) / std


def extract_layers(
    model: torch.nn.Module,
    image: torch.Tensor,
    layer_ids: list[int],
    input_size: int,
) -> list[torch.Tensor]:
    features, final = model.forward_with_features(preprocess_dinov2(image, input_size))
    if features:
        features[max(features)] = final
    missing = sorted(set(layer_ids) - set(features))
    if missing:
        raise ValueError(f"DINOv2 layers {missing} are unavailable; choose from {sorted(features)}")
    return [features[index] for index in layer_ids]
