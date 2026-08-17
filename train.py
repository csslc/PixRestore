#!/usr/bin/env python
"""Train PixRestore on paired LQ/GT manifests."""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

from pixrestore import LightningDiT_PixelDiffusion, PixelDiffusion
from pixrestore.data import PairedJsonlDataset, collate_pairs
from pixrestore.gan import MultiLayerDinoDiscriminator, as_feature_list, set_requires_grad
from pixrestore.vision import extract_layers, load_dinov2

LOGGER = logging.getLogger("pixrestore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--manifest", nargs="+", help="Override training manifests")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--test-lq-dir", help="Override evaluation LQ directory")
    parser.add_argument("--test-gt-dir", help="Override evaluation GT directory")
    parser.add_argument("--pretrained", help="Model weights or a checkpoint directory")
    parser.add_argument("--dinov2-repository", help="Optional local DINOv2 repository clone")
    parser.add_argument("--dinov2-checkpoint", help="Optional local DINOv2 checkpoint")
    parser.add_argument("--max-steps", type=int, help="Override the number of training steps")
    parser.add_argument("--checkpoint-steps", type=int, help="Override checkpoint frequency")
    parser.add_argument("--eval-steps", type=int, help="Override evaluation frequency")
    parser.add_argument("--global-batch-size", type=int, help="Override global batch size")
    parser.add_argument("--num-workers", type=int, help="Override data loader workers")
    parser.add_argument(
        "--resolution",
        type=int,
        help="Override training crop / model resolution (must be divisible by patch_size)",
    )
    return parser.parse_args()


def load_config(cli: argparse.Namespace) -> SimpleNamespace:
    with Path(cli.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    base_config = config.pop("base_config", None)
    if base_config:
        with Path(base_config).open(encoding="utf-8") as stream:
            merged = yaml.safe_load(stream)
        merged.update(config)
        config = merged
    overrides = {
        "manifests": cli.manifest,
        "output_dir": cli.output_dir,
        "test_lq_dir": cli.test_lq_dir,
        "test_gt_dir": cli.test_gt_dir,
        "pretrained": cli.pretrained,
        "dinov2_repository": cli.dinov2_repository,
        "dinov2_checkpoint": cli.dinov2_checkpoint,
        "max_steps": cli.max_steps,
        "checkpoint_steps": cli.checkpoint_steps,
        "eval_steps": cli.eval_steps,
        "global_batch_size": cli.global_batch_size,
        "num_workers": cli.num_workers,
        "resolution": cli.resolution,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
    if not config.get("manifests"):
        raise ValueError("Set manifests in the config or pass --manifest")
    resolution = int(config["resolution"])
    patch_size = int(config["patch_size"])
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    if resolution % patch_size:
        raise ValueError(
            f"resolution {resolution} must be divisible by patch_size {patch_size}"
        )
    return SimpleNamespace(**config)


def build_model(config: SimpleNamespace) -> LightningDiT_PixelDiffusion:
    return LightningDiT_PixelDiffusion(
        input_size=config.resolution,
        patch_size=config.patch_size,
        in_channels=6,
        out_channels=3,
        hidden_size=config.hidden_size,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        z_dims=config.encoder_dim,
        encdim_ratio=config.encoder_dim_ratio,
        use_qknorm=config.use_qknorm,
        use_swiglu=config.use_swiglu,
        use_rope=config.use_rope,
        use_rmsnorm=config.use_rmsnorm,
        adain_single=True,
        num_fused_layers=len(config.encoder_layers),
        pca_dim=config.bottleneck_dim,
        use_bottleneck_patch_embed=True,
        use_dino_layer_router=config.use_dino_layer_router,
        dino_layer_router_mode=config.dino_layer_router_mode,
        same_condition_layer_weights=config.same_condition_layer_weights,
        gate_temperature=config.gate_temperature,
        feature_norm=config.feature_norm,
        use_lq_degradation_token=bool(getattr(config, "use_lq_degradation_token", False)),
        register_local_gate_logit_scale=bool(
            getattr(config, "register_local_gate_logit_scale", False)
        ),
    )


def build_flow(config: SimpleNamespace, accelerator: Accelerator) -> PixelDiffusion:
    return PixelDiffusion(
        flow_ratio=config.flow_ratio,
        time_dist=config.time_distribution,
        alpha=config.alpha,
        z_start="noise",
        cfg_ratio=config.cfg_ratio,
        cfg_scale=config.cfg_scale,
        image_size=config.resolution,
        channels=3,
        interp_type="lin",
        uncond_type="zero",
        norm_p=config.adaptive_loss_power,
        accelerator=accelerator,
        t_start=0,
        t_end=1,
        use_cos=False,
        args=config,
    )


@torch.no_grad()
def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    source = dict(model.named_parameters())
    for name, parameter in ema.named_parameters():
        parameter.mul_(decay).add_(source[name].detach(), alpha=1 - decay)


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = sorted(
        output_dir.glob("checkpoints/checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    return checkpoints[-1] if checkpoints else None


def load_pretrained(
    model: torch.nn.Module,
    ema: torch.nn.Module,
    path: str | Path,
) -> None:
    path = Path(path).expanduser()
    if path.is_dir():
        candidates = [
            path / "ema_model.safetensors",
            path / "clean_weights" / "ema_model.safetensors",
            path / "model.safetensors",
            path / "clean_weights" / "model.safetensors",
        ]
        path = next((candidate for candidate in candidates if candidate.is_file()), path)
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained weights not found: {path}")
    state = load_file(path) if path.suffix == ".safetensors" else torch.load(path, map_location="cpu")
    current = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state.items():
        if key in current and tuple(current[key].shape) != tuple(value.shape):
            skipped.append(f"{key}: ckpt {tuple(value.shape)} vs model {tuple(current[key].shape)}")
            continue
        filtered[key] = value
    if skipped:
        LOGGER.warning(
            "Skipped %d pretrained keys with mismatched shapes (common when changing resolution): %s",
            len(skipped),
            skipped,
        )
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    skipped_keys = {item.split(":", 1)[0] for item in skipped}
    missing = [key for key in missing if key not in skipped_keys]
    if unexpected:
        LOGGER.warning("Ignored %d unexpected pretrained keys", len(unexpected))
    if missing:
        LOGGER.warning("Pretrained checkpoint is missing %d model keys", len(missing))
    ema.load_state_dict(model.state_dict())


def save_preview(batch: dict, output_dir: Path) -> None:
    preview = output_dir / "first_batch"
    preview.mkdir(parents=True, exist_ok=True)
    lq = batch["lq"].float().cpu().add(1).mul(0.5)
    hq = batch["hq"].float().cpu().add(1).mul(0.5)
    save_image(lq, preview / "lq.png", nrow=min(8, len(lq)))
    save_image(hq, preview / "gt.png", nrow=min(8, len(hq)))
    with (preview / "pairs.jsonl").open("w", encoding="utf-8") as stream:
        for lq_path, gt_path in zip(batch["lq_path"], batch["gt_path"]):
            stream.write(json.dumps({"lq": lq_path, "gt": gt_path}, ensure_ascii=False) + "\n")


def center_crop(image, size: int):
    width, height = image.size
    scale = size / min(width, height)
    image = image.resize((round(width * scale), round(height * scale)))
    left = (image.width - size) // 2
    top = (image.height - size) // 2
    return image.crop((left, top, left + size, top + size))


@torch.no_grad()
def evaluate(
    config: SimpleNamespace,
    accelerator: Accelerator,
    model: torch.nn.Module,
    encoder: torch.nn.Module,
    flow: PixelDiffusion,
    step: int,
) -> None:
    if not config.test_lq_dir:
        return
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor

    paths = sorted(
        path
        for path in Path(config.test_lq_dir).iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    paths = paths[accelerator.process_index :: accelerator.num_processes]
    output_dir = Path(config.output_dir) / "samples" / f"{step:08d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_paths = {}
    if config.test_gt_dir:
        gt_paths = {
            path.stem: path
            for path in Path(config.test_gt_dir).iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        }
    psnr_sum = torch.zeros(1, device=accelerator.device)
    psnr_count = torch.zeros(1, device=accelerator.device)
    model.eval()
    for path in paths:
        with Image.open(path) as image:
            image = center_crop(image.convert("RGB"), config.resolution)
        lq = pil_to_tensor(image).unsqueeze(0).to(accelerator.device).float().div_(127.5).sub_(1)
        with accelerator.autocast():
            features = extract_layers(encoder, lq, config.encoder_layers, config.encoder_input_size)
            restored = flow.sample_multistep_fm(
                model,
                lq,
                venc_fea=features,
                n_steps=config.eval_sampling_steps,
            )
        restored = restored.add(1).mul(0.5).clamp(0, 1)
        save_image(restored, output_dir / f"{path.stem}.png")
        if path.stem in gt_paths:
            with Image.open(gt_paths[path.stem]) as image:
                gt = center_crop(image.convert("RGB"), config.resolution)
            gt = pil_to_tensor(gt).unsqueeze(0).to(accelerator.device).float().div_(255)
            mse = (restored.float() - gt).square().mean().clamp_min(1e-12)
            psnr_sum += -10 * torch.log10(mse)
            psnr_count += 1
    psnr_sum = accelerator.reduce(psnr_sum, reduction="sum")
    psnr_count = accelerator.reduce(psnr_count, reduction="sum")
    if accelerator.is_main_process and psnr_count.item():
        psnr = (psnr_sum / psnr_count).item()
        LOGGER.info("Evaluation at step %d: PSNR %.4f dB", step, psnr)
        if config.report_to:
            accelerator.log({"eval/psnr": psnr}, step=step)
    model.train()


def main() -> None:
    config = load_config(parse_args())
    output_dir = Path(config.output_dir)
    project = ProjectConfiguration(project_dir=output_dir, logging_dir=output_dir / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.report_to or None,
        project_config=project,
    )
    logging.basicConfig(
        level=logging.INFO if accelerator.is_local_main_process else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    set_seed(config.seed)

    local_batch_size = config.global_batch_size // accelerator.num_processes
    if config.global_batch_size % accelerator.num_processes:
        raise ValueError("global_batch_size must be divisible by the number of processes")
    dataset = PairedJsonlDataset(config.manifests, config.resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
        drop_last=True,
        collate_fn=collate_pairs,
    )

    model = build_model(config)
    ema = copy.deepcopy(model).to(accelerator.device).eval().requires_grad_(False)
    discriminator = None
    if config.use_dino_gan:
        discriminator = MultiLayerDinoDiscriminator(
            feature_dim=config.encoder_dim,
            num_layers=len(config.encoder_layers),
            hidden_ratio=config.dino_gan_hidden_ratio,
            max_hidden=config.dino_gan_max_hidden,
            real_label=config.dino_gan_real_label,
        ).to(accelerator.device)
    encoder = load_dinov2(
        config.encoder_type,
        accelerator.device,
        repository=config.dinov2_repository,
        checkpoint=config.dinov2_checkpoint,
    )
    flow = build_flow(config, accelerator)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=tuple(config.adam_betas),
        weight_decay=config.weight_decay,
        eps=config.adam_epsilon,
    )
    discriminator_optimizer = None
    if discriminator is not None:
        discriminator_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=config.dino_gan_learning_rate,
            betas=tuple(config.adam_betas),
            weight_decay=config.weight_decay,
            eps=config.adam_epsilon,
        )
    scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps,
    )
    discriminator_scheduler = None
    if discriminator_optimizer is not None:
        discriminator_scheduler = get_scheduler(
            config.lr_scheduler,
            discriminator_optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_steps,
        )
    accelerator.register_for_checkpointing(ema)
    if discriminator is None:
        model, optimizer, dataloader, scheduler = accelerator.prepare(
            model, optimizer, dataloader, scheduler
        )
    else:
        (
            model,
            discriminator,
            optimizer,
            discriminator_optimizer,
            dataloader,
            scheduler,
            discriminator_scheduler,
        ) = accelerator.prepare(
            model,
            discriminator,
            optimizer,
            discriminator_optimizer,
            dataloader,
            scheduler,
            discriminator_scheduler,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        with (output_dir / "config.json").open("w") as stream:
            json.dump(vars(config), stream, indent=2)
        if config.report_to:
            accelerator.init_trackers(config.project_name, config=vars(config))

    step = 0
    resume = latest_checkpoint(output_dir) if config.resume == "latest" else None
    if config.resume and config.resume != "latest":
        resume = Path(config.resume)
    if resume:
        accelerator.load_state(resume)
        step = int(resume.name.rsplit("-", 1)[-1])
        LOGGER.info("Resumed from %s", resume)
    elif config.pretrained:
        load_pretrained(accelerator.unwrap_model(model), ema, config.pretrained)
        LOGGER.info("Loaded pretrained model from %s", config.pretrained)

    progress = tqdm(
        total=config.max_steps,
        initial=step,
        disable=not accelerator.is_local_main_process,
        desc="Training",
    )
    preview_saved = False
    while step < config.max_steps:
        for batch in dataloader:
            if not preview_saved:
                if accelerator.is_main_process and config.save_first_batch:
                    save_preview(batch, output_dir)
                preview_saved = True

            lq = batch["lq"].to(accelerator.device, non_blocking=True)
            hq = batch["hq"].to(accelerator.device, non_blocking=True)
            invalid_mask = batch.get("invalid_mask")
            if invalid_mask is not None:
                invalid_mask = invalid_mask.to(accelerator.device, non_blocking=True)

            with torch.no_grad(), accelerator.autocast():
                lq_features = extract_layers(
                    encoder, lq, config.encoder_layers, config.encoder_input_size
                )
                hq_features = extract_layers(
                    encoder, hq, config.encoder_layers, config.encoder_input_size
                )

            accumulate_models = (model,) if discriminator is None else (model, discriminator)
            with accelerator.accumulate(*accumulate_models):
                fake_features = None

                def dino_loss_fn(image):
                    nonlocal fake_features
                    fake_features = extract_layers(
                        encoder, image, config.encoder_layers, config.encoder_input_size
                    )
                    return fake_features

                loss, backward_loss, _ = flow.loss_fm(
                    model,
                    lq,
                    hq,
                    lq_features,
                    invalid_mask=invalid_mask,
                    z_hq=hq_features,
                    dino_loss_fn=dino_loss_fn,
                )
                generator_gan_loss = None
                discriminator_real_loss = None
                discriminator_fake_loss = None
                if discriminator is not None:
                    if fake_features is None:
                        raise RuntimeError("DINO-GAN requires use_hierar_loss=true")
                    set_requires_grad(discriminator, False)
                    generator_gan_loss = (
                        discriminator(fake_features, real=True) * config.dino_gan_loss_weight
                    )
                    backward_loss = backward_loss + generator_gan_loss

                accelerator.backward(backward_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if discriminator is not None:
                    set_requires_grad(discriminator, True)
                    discriminator_fake_loss = (
                        discriminator(
                            [feature.detach() for feature in as_feature_list(fake_features)],
                            real=False,
                        )
                        * config.dino_gan_loss_weight
                    )
                    discriminator_real_loss = (
                        discriminator(
                            [feature.detach() for feature in as_feature_list(hq_features)],
                            real=True,
                        )
                        * config.dino_gan_loss_weight
                    )
                    accelerator.backward(discriminator_fake_loss + discriminator_real_loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            discriminator.parameters(), config.max_grad_norm
                        )
                    discriminator_optimizer.step()
                    discriminator_scheduler.step()
                    discriminator_optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue
            step += 1
            progress.update(1)
            update_ema(ema, accelerator.unwrap_model(model), config.ema_decay)
            logs = {"loss": loss.detach(), "lr": scheduler.get_last_lr()[0]}
            if flow.last_hierar_loss is not None:
                logs["hierar_loss"] = flow.last_hierar_loss.detach()
            if generator_gan_loss is not None:
                logs.update(
                    {
                        "dino_gan/generator": generator_gan_loss.detach(),
                        "dino_gan/discriminator_real": discriminator_real_loss.detach(),
                        "dino_gan/discriminator_fake": discriminator_fake_loss.detach(),
                    }
                )
            progress.set_postfix(loss=f"{loss.detach().item():.4f}")
            if config.report_to:
                accelerator.log(logs, step=step)

            if step % config.checkpoint_steps == 0:
                checkpoint = output_dir / "checkpoints" / f"checkpoint-{step:08d}"
                accelerator.save_state(checkpoint)
                if accelerator.is_main_process:
                    weights = {
                        name: value.detach().cpu()
                        for name, value in ema.state_dict().items()
                    }
                    save_file(weights, checkpoint / "ema_model.safetensors")
                    if discriminator is not None:
                        discriminator_weights = {
                            name: value.detach().cpu()
                            for name, value in accelerator.unwrap_model(
                                discriminator
                            ).state_dict().items()
                        }
                        save_file(
                            discriminator_weights,
                            checkpoint / "dino_gan_discriminator.safetensors",
                        )

            if config.eval_steps and step % config.eval_steps == 0:
                evaluate(config, accelerator, ema, encoder, flow, step)
                accelerator.wait_for_everyone()
            if step >= config.max_steps:
                break

    progress.close()
    accelerator.end_training()


if __name__ == "__main__":
    main()
