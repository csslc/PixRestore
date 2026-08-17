#!/usr/bin/env python
"""Run PixRestore inference on one image, a folder, or a JSON/JSONL manifest."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from accelerate import Accelerator
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from tqdm.auto import tqdm

from pixrestore import LightningDiT_PixelDiffusion, PixelDiffusion
from pixrestore.vision import extract_layers, load_dinov2


LOGGER = logging.getLogger("pixrestore.inference")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--checkpoint", required=True, help="Checkpoint directory or weight file")
    parser.add_argument("-i", "--input", required=True, help="Image, image folder, JSON, or JSONL")
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument(
        "--config",
        help="Optional training YAML/JSON override. Default: config.json next to the checkpoint.",
    )
    parser.add_argument("--infer-steps", "--infer_steps", type=int, default=10)
    parser.add_argument("--cfg-scale", "--cfg_scale", type=float)
    parser.add_argument(
        "--cond-strength-aelq-test",
        "--cond_strength_aelq_test",
        type=float,
    )
    parser.add_argument(
        "--weak-cond-strength-aelq",
        "--weak_cond_strength_aelq",
        type=float,
    )
    parser.add_argument("--schedule", choices=("linear", "cosine"), default="linear")
    parser.add_argument("--method", help="Output method name for JSON/JSONL input")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--upscale", type=float, default=1.0)
    parser.add_argument(
        "--resize-to-train",
        "--resize_to_train",
        action="store_true",
        help="Resize every input to the training resolution.",
    )
    parser.add_argument(
        "--test-mode",
        "--test_mode",
        choices=("off", "resize", "center_crop"),
        default="off",
        help="Test-time preprocessing: resize (short-edge to resolution + center crop), "
        "center_crop (512 then scale to resolution), or off.",
    )
    parser.add_argument("--dinov2-repository", help="Local DINOv2 repository")
    parser.add_argument("--dinov2-checkpoint", help="Local DINOv2 weights")
    return parser.parse_args()


def checkpoint_directory(path: Path) -> Path:
    return path if path.is_dir() else path.parent


def find_config_path(checkpoint: Path, explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    start = checkpoint_directory(checkpoint)
    # Prefer the pretrained/checkpoint directory itself, then walk parents.
    search_dirs = [start]
    if start.name == "clean_weights":
        search_dirs.append(start.parent)
    search_dirs.extend(start.parents)
    seen: set[Path] = set()
    for directory in search_dirs:
        directory = directory.resolve()
        if directory in seen:
            continue
        seen.add(directory)
        candidate = directory / "config.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Cannot find config.json under {start}. "
        "Place config.json next to the pretrained weights, or pass --config."
    )


def read_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream) if path.suffix.lower() == ".json" else yaml.safe_load(stream)
    base_config = config.pop("base_config", None)
    if base_config:
        base_path = Path(base_config)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = read_config(base_path.resolve())
        merged.update(config)
        config = merged
    return config


def read_state_dict(path: Path) -> dict:
    if path.suffix == ".safetensors":
        state = load_file(str(path))
    else:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


def apply_checkpoint_architecture_flags(config_data: dict, state: dict) -> dict:
    """Infer LQ-degradation gate flags from checkpoint weights when possible."""
    has_lq_deg = any(key.startswith("lq_degradation_encoder.") for key in state)
    gate_weight = state.get("layer_gates.weight")
    num_fused = len(config_data.get("encoder_layers") or [])
    hidden = int(config_data.get("hidden_size") or 0)
    if gate_weight is not None and hidden > 0 and num_fused > 1:
        expected_with_deg = hidden * (num_fused + 4)
        if int(gate_weight.shape[1]) == expected_with_deg:
            has_lq_deg = True
    if has_lq_deg:
        config_data["use_lq_degradation_token"] = True
    config_data["register_local_gate_logit_scale"] = "local_gate_logit_scale" in state
    return config_data


def build_model(config: SimpleNamespace) -> LightningDiT_PixelDiffusion:
    """Build exactly the architecture used by train.py."""
    return LightningDiT_PixelDiffusion(
        input_size=config.resolution,
        patch_size=config.patch_size,
        in_channels=6,
        out_channels=3,
        hidden_size=config.hidden_size,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        z_dims=config.encoder_dim if config.use_venc else None,
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


def find_weights(checkpoint: Path) -> Path:
    if checkpoint.is_file():
        return checkpoint
    candidates = (
        checkpoint / "ema_model.safetensors",
        checkpoint / "clean_weights" / "ema_model.safetensors",
        checkpoint / "model.safetensors",
        checkpoint / "clean_weights" / "model.safetensors",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No model weights found under {checkpoint}")


def load_weights(model: torch.nn.Module, path: Path, state: dict | None = None) -> None:
    if state is None:
        state = read_state_dict(path)
    model.load_state_dict(state, strict=True)


def safe_name(value: object, default: str) -> str:
    text = str(value or default).strip()
    return text.replace("/", "_").replace("\\", "_")


def read_manifest(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Input manifest is empty: {path}")
    rows = json.loads(text) if text.startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()
    ]
    samples = []
    for row in rows:
        image_path = Path(row.get("lq") or row.get("LQ") or "")
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        if not str(row.get("lq") or row.get("LQ") or "").strip():
            raise ValueError(f"Manifest row has no lq field: {row}")
        samples.append(
            {
                "path": image_path.resolve(),
                "type": safe_name(row.get("type"), "unknown_type"),
                "data": safe_name(row.get("data"), "unknown_data"),
                "method": safe_name(row.get("method"), ""),
                "manifest": True,
            }
        )
    return samples


def collect_samples(input_path: Path) -> list[dict]:
    if input_path.is_dir():
        paths = sorted(
            path for path in input_path.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        samples = [{"path": path.resolve(), "manifest": False} for path in paths]
    elif input_path.suffix.lower() in {".json", ".jsonl"}:
        samples = read_manifest(input_path)
    else:
        samples = [{"path": input_path.resolve(), "manifest": False}]
    if not samples:
        raise FileNotFoundError(f"No input images found in {input_path}")
    for index, sample in enumerate(samples):
        sample["index"] = index
    return samples


def resize_short_edge_and_crop(image: Image.Image, size: int) -> Image.Image:
    scale = size / min(image.size)
    resized = image.resize(
        (round(image.width * scale), round(image.height * scale)),
        Image.Resampling.BICUBIC,
    )
    left = (resized.width - size) // 2
    top = (resized.height - size) // 2
    return resized.crop((left, top, left + size, top + size))


def crop_to_patch_multiple(image: Image.Image, patch_size: int) -> Image.Image:
    width = image.width - image.width % patch_size
    height = image.height - image.height % patch_size
    if width < patch_size or height < patch_size:
        raise ValueError(f"Image {image.size} is smaller than patch size {patch_size}")
    left = (image.width - width) // 2
    top = (image.height - height) // 2
    return image.crop((left, top, left + width, top + height))


def preprocess_image(image: Image.Image, args: argparse.Namespace, config: SimpleNamespace) -> Image.Image:
    image = image.convert("RGB")
    if args.upscale != 1:
        image = image.resize(
            (round(image.width * args.upscale), round(image.height * args.upscale)),
            Image.Resampling.BICUBIC,
        )
    if args.test_mode == "resize":
        return resize_short_edge_and_crop(image, config.resolution)
    if args.test_mode == "center_crop":
        image = resize_short_edge_and_crop(image, 512)
        if config.resolution != 512:
            image = image.resize(
                (config.resolution, config.resolution), Image.Resampling.BICUBIC
            )
        return image
    if args.resize_to_train:
        return image.resize(
            (config.resolution, config.resolution), Image.Resampling.BICUBIC
        )
    return crop_to_patch_multiple(image, config.patch_size)


def output_path(
    sample: dict,
    output_root: Path,
    method: str | None,
    config: SimpleNamespace,
    infer_steps: int,
    schedule: str,
) -> Path:
    if sample["manifest"]:
        name = safe_name(method or sample["method"], "pixrestore")
        folder = output_root / sample["type"] / sample["data"] / (
            f"{name}_cfg{config.cfg_scale:g}_step{infer_steps}"
        )
    else:
        folder = output_root / (
            f"steps{infer_steps}_cfg{config.cfg_scale:g}_sched{schedule}"
        )
    return folder / f"{sample['path'].stem}.png"


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    config_path = find_config_path(checkpoint, args.config)
    config_data = read_config(config_path)

    if args.cfg_scale is not None:
        config_data["cfg_scale"] = args.cfg_scale
    if args.cond_strength_aelq_test is not None:
        config_data["cond_strength_aelq_test"] = args.cond_strength_aelq_test
    if args.weak_cond_strength_aelq is not None:
        value = args.weak_cond_strength_aelq
        config_data["weak_cond_strength_aelq_list"] = [value, value]
    if args.dinov2_repository:
        config_data["dinov2_repository"] = args.dinov2_repository
    if args.dinov2_checkpoint:
        config_data["dinov2_checkpoint"] = args.dinov2_checkpoint
    weight_path = find_weights(checkpoint)
    state = read_state_dict(weight_path)
    config_data = apply_checkpoint_architecture_flags(config_data, state)
    config = SimpleNamespace(**config_data)

    accelerator = Accelerator(mixed_precision=str(config.mixed_precision or "no"))
    logging.basicConfig(
        level=logging.INFO if accelerator.is_local_main_process else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    device = accelerator.device

    model = build_model(config)
    load_weights(model, weight_path, state=state)
    model.to(device).eval().requires_grad_(False)
    flow = build_flow(config, accelerator)
    encoder = None
    if config.use_venc:
        encoder = load_dinov2(
            config.encoder_type,
            device,
            repository=config.dinov2_repository,
            checkpoint=config.dinov2_checkpoint,
        )

    samples = collect_samples(Path(args.input).expanduser().resolve())
    local_samples = samples[accelerator.process_index :: accelerator.num_processes]
    output_root = Path(args.output_dir).expanduser().resolve()
    if accelerator.is_main_process:
        LOGGER.info("Config: %s", config_path)
        LOGGER.info("Weights: %s", weight_path)
        LOGGER.info("Images: %d across %d process(es)", len(samples), accelerator.num_processes)

    progress = tqdm(local_samples, disable=not accelerator.is_local_main_process)
    for sample in progress:
        if not sample["path"].is_file():
            raise FileNotFoundError(f"Input image does not exist: {sample['path']}")
        torch.manual_seed(args.seed + sample["index"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + sample["index"])

        with Image.open(sample["path"]) as image:
            image = preprocess_image(image, args, config)
        lq = pil_to_tensor(image).unsqueeze(0).to(device).float().div_(127.5).sub_(1)

        with torch.no_grad(), accelerator.autocast():
            features = (
                extract_layers(
                    encoder,
                    lq,
                    config.encoder_layers,
                    config.encoder_input_size,
                )
                if encoder is not None
                else None
            )
            restored = flow.sample_multistep_fm(
                model,
                lq,
                venc_fea=features,
                n_steps=args.infer_steps,
                schedule=args.schedule,
            )

        destination = output_path(
            sample,
            output_root,
            args.method,
            config,
            args.infer_steps,
            args.schedule,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        image = restored[0].float().cpu().add(1).mul(0.5).clamp(0, 1)
        to_pil_image(image).save(destination)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        LOGGER.info("Done. Results saved under %s", output_root)


if __name__ == "__main__":
    main()
