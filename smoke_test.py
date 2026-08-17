#!/usr/bin/env python
"""Fast, self-contained checks for the standard and DINO-GAN training paths."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pixrestore import LightningDiT_PixelDiffusion, MeanFlowIR_PixelDiffusion
from pixrestore.data import PairedJsonlDataset, collate_pairs
from pixrestore.gan import MultiLayerDinoDiscriminator, as_feature_list, set_requires_grad


class TinyFrozenEncoder(nn.Module):
    def __init__(self, feature_dim: int, num_layers: int) -> None:
        super().__init__()
        self.projection = nn.Linear(3, feature_dim, bias=False)
        self.num_layers = num_layers
        self.requires_grad_(False)

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        tokens = F.adaptive_avg_pool2d(image, (2, 2)).flatten(2).transpose(1, 2)
        tokens = self.projection(tokens)
        return [tokens * (1 + 0.1 * index) for index in range(self.num_layers)]


def make_batch() -> dict:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        Image.new("RGB", (24, 20), (64, 96, 128)).save(root / "lq.png")
        Image.new("RGB", (24, 20), (96, 128, 160)).save(root / "gt.png")
        manifest = root / "pairs.jsonl"
        manifest.write_text(
            json.dumps({"lq": str(root / "lq.png"), "gt": str(root / "gt.png")}) + "\n"
        )
        dataset = PairedJsonlDataset([manifest], crop_size=16)
        return next(iter(DataLoader(dataset, batch_size=1, collate_fn=collate_pairs)))


def make_config(mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        use_time_shift=False,
        time_shift_base_dimension=4096,
        cond_strength_aelq_list=[5.0, 1.0],
        aug_noise=False,
        use_aelq=True,
        use_venc=True,
        weak_cond_strength_aelq_list=[0.99, 1.0],
        weak_cond_strength_venc=0.0,
        cond_strength_venc=1.0,
        cond_strength_aelq_test=1.0,
        t_eps=0.05,
        fixed_train_t=1.0 if mode == "gan" else None,
        dino_layer_router_mode="content",
        use_dino_layer_router=True,
        dino_layer_router_loss_weight=0.5,
        use_hierar_loss=True,
        dino_hierar_loss_type="cosine",
        dino_hierar_loss_weight=0.5,
        average_hierar_loss_weights=False,
        same_condition_layer_weights=False,
        gate_temperature=1.0,
    )


def run(mode: str) -> None:
    torch.manual_seed(0)
    config = make_config(mode)
    feature_dim = 24
    num_layers = 3
    encoder = TinyFrozenEncoder(feature_dim, num_layers)
    model = LightningDiT_PixelDiffusion(
        input_size=16,
        patch_size=8,
        in_channels=6,
        out_channels=3,
        hidden_size=48,
        depth=2,
        num_heads=6,
        mlp_ratio=2,
        z_dims=feature_dim,
        encdim_ratio=2,
        num_fused_layers=num_layers,
        pca_dim=24,
        use_qknorm=True,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        use_dino_layer_router=True,
        dino_layer_router_mode="content",
        same_condition_layer_weights=False,
        gate_temperature=1.0,
        feature_norm="channel_rmsnorm",
    )
    flow = MeanFlowIR_PixelDiffusion(
        accelerator=SimpleNamespace(device=torch.device("cpu")),
        args=config,
        cfg_ratio=0.1,
    )
    batch = make_batch()
    lq, hq = batch["lq"], batch["hq"]
    lq_features = encoder(lq)
    hq_features = encoder(hq)
    fake_features = None

    def feature_loss(image):
        nonlocal fake_features
        fake_features = encoder(image)
        return fake_features

    loss, backward_loss, _ = flow.loss_fm(
        model,
        lq,
        hq,
        lq_features,
        z_hq=hq_features,
        dino_loss_fn=feature_loss,
    )

    discriminator = None
    if mode == "gan":
        discriminator = MultiLayerDinoDiscriminator(feature_dim, num_layers)
        set_requires_grad(discriminator, False)
        generator_gan_loss = discriminator(fake_features, real=True) * 0.5
        backward_loss = backward_loss + generator_gan_loss

    backward_loss.backward()
    if not any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Model received no gradients")

    if discriminator is not None:
        set_requires_grad(discriminator, True)
        discriminator.zero_grad(set_to_none=True)
        fake_loss = discriminator(
            [feature.detach() for feature in as_feature_list(fake_features)], real=False
        )
        real_loss = discriminator(
            [feature.detach() for feature in as_feature_list(hq_features)], real=True
        )
        (fake_loss + real_loss).backward()
        if not any(parameter.grad is not None for parameter in discriminator.parameters()):
            raise RuntimeError("Discriminator received no gradients")

    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss: {loss}")
    print(f"OK: {mode} training path, loss={loss.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("standard", "gan"), required=True)
    run(parser.parse_args().mode)
