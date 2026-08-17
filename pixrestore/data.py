"""Paired LQ/GT datasets described by JSON Lines manifests."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


def _read_manifest(path: str | Path) -> list[tuple[str, str, str | None]]:
    pairs = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            lq = row.get("lq") or row.get("LQ")
            gt = row.get("gt") or row.get("GT")
            mask = row.get("invalid_mask") or row.get("INVALID_MASK")
            if lq and gt:
                pairs.append((str(lq), str(gt), str(mask) if mask else None))
    if not pairs:
        raise ValueError(f"No valid LQ/GT pairs found in {path}")
    return pairs


def _resize_short_edge(image: Image.Image, edge: int, resample: int) -> Image.Image:
    if min(image.size) >= edge:
        return image
    scale = edge / min(image.size)
    size = tuple(round(value * scale) for value in image.size)
    return image.resize(size, resample)


def _paired_random_crop(
    lq: Image.Image,
    gt: Image.Image,
    edge: int,
    mask: Image.Image | None,
) -> tuple[Image.Image, Image.Image, Image.Image | None]:
    if gt.size != lq.size:
        gt = gt.resize(lq.size, Image.Resampling.LANCZOS)
    if mask is not None and mask.size != lq.size:
        mask = mask.resize(lq.size, Image.Resampling.NEAREST)

    lq = _resize_short_edge(lq, edge, Image.Resampling.BICUBIC)
    gt = gt.resize(lq.size, Image.Resampling.LANCZOS)
    if mask is not None:
        mask = mask.resize(lq.size, Image.Resampling.NEAREST)

    left = random.randint(0, lq.width - edge)
    top = random.randint(0, lq.height - edge)
    box = (left, top, left + edge, top + edge)
    return lq.crop(box), gt.crop(box), mask.crop(box) if mask else None


def _load_mask(path: str | None) -> Image.Image | None:
    if path is None:
        return None
    mask = np.squeeze(np.load(path))
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D invalid mask, got {mask.shape}")
    return Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")


def _to_tensor(image: Image.Image) -> torch.Tensor:
    return pil_to_tensor(image).float().div_(127.5).sub_(1.0)


class PairedJsonlDataset(Dataset):
    """Samples manifests equally, then applies aligned LQ/GT random crops.

    Manifest rows use ``{"lq": "...", "gt": "..."}`` and may include an
    ``invalid_mask`` NumPy file.
    """

    def __init__(
        self,
        manifests: Iterable[str | Path],
        crop_size: int,
        *,
        max_retries: int = 50,
    ) -> None:
        self.groups = [_read_manifest(path) for path in manifests]
        if not self.groups:
            raise ValueError("At least one training manifest is required")
        self.crop_size = int(crop_size)
        self.max_retries = int(max_retries)
        self.has_masks = any(mask for group in self.groups for _, _, mask in group)
        self.group_length = max(map(len, self.groups))

    def __len__(self) -> int:
        return self.group_length * len(self.groups)

    def __getitem__(self, index: int) -> dict[str, Any]:
        group_index = index % len(self.groups)
        group = self.groups[group_index]
        pair_index = (index // len(self.groups)) % len(group)
        last_error: Exception | None = None

        for retry in range(self.max_retries):
            lq_path, gt_path, mask_path = group[(pair_index + retry) % len(group)]
            try:
                with Image.open(lq_path) as image:
                    lq = image.convert("RGB")
                with Image.open(gt_path) as image:
                    gt = image.convert("RGB")
                mask = _load_mask(mask_path)
                lq, gt, mask = _paired_random_crop(lq, gt, self.crop_size, mask)

                sample = {
                    "lq": _to_tensor(lq),
                    "hq": _to_tensor(gt),
                    "lq_path": lq_path,
                    "gt_path": gt_path,
                }
                if self.has_masks:
                    sample["invalid_mask"] = (
                        pil_to_tensor(mask).float().div_(255.0)
                        if mask is not None
                        else torch.zeros(1, self.crop_size, self.crop_size)
                    )
                return sample
            except (OSError, ValueError, RuntimeError) as error:
                last_error = error
        raise RuntimeError(f"Failed to load pair after {self.max_retries} attempts") from last_error


def collate_pairs(batch: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "lq": torch.stack([sample["lq"] for sample in batch]),
        "hq": torch.stack([sample["hq"] for sample in batch]),
        "lq_path": [sample["lq_path"] for sample in batch],
        "gt_path": [sample["gt_path"] for sample in batch],
    }
    if "invalid_mask" in batch[0]:
        result["invalid_mask"] = torch.stack([sample["invalid_mask"] for sample in batch])
    return result
