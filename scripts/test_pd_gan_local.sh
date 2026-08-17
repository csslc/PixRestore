#!/usr/bin/env bash
# Continue local_pd14_test-ppu4 (DiT-S + DINOv2-S) with DINO-GAN finetuning.
# Architecture is pinned in configs/train_gan.yaml so it does not inherit DiT-B.
set -euo pipefail

cd /home/notebook/data/group/slc/pixrestore

SOURCE_ROOT=/home/notebook/data/group/slc/pisa-sr

MANIFESTS=(
  "$SOURCE_ROOT/dataloaders/slc_5_3_uav_sync/all.json"
  "$SOURCE_ROOT/dataloaders/slc_7_7_short_data/deblur.json"
  "$SOURCE_ROOT/dataloaders/slc_7_7_short_data/dehaze_6_15_all.json"
  "$SOURCE_ROOT/dataloaders/slc_7_7_short_data/denoise.json"
  "$SOURCE_ROOT/dataloaders/slc_7_7_short_data/derain_6_25_woblur.json"
  "$SOURCE_ROOT/dataloaders/slc_7_7_short_data/desnow.json"
  "$SOURCE_ROOT/dataloaders/slc_7_7_short_data/low_light_enhancement_6_25_all.json"
  "$SOURCE_ROOT/dataloaders/slc_7_7_short_data/realesrgan.json"
)

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
XFORMERS_DISABLED=1 \
TORCHDYNAMO_DISABLE=1 \
TORCH_COMPILE_DISABLE=1 \
accelerate launch \
  --multi_gpu \
  --num_processes 8 \
  --main_process_port 29528 \
  train.py \
  --config configs/train_gan.yaml \
  --manifest "${MANIFESTS[@]}" \
  --test-lq-dir /home/notebook/data/group/slc/Datasets/IR_testdata/selected_testsets/test_a_little/LQ \
  --test-gt-dir /home/notebook/data/group/slc/Datasets/IR_testdata/selected_testsets/test_a_little/GT \
  --output-dir outputs/local_pd_gan_test-250k-ppu5 \
  --dinov2-checkpoint /home/notebook/data/group/slc/pisa-sr/pretrain_models/pisasr_preview/torch_cache/checkpoints/dinov2_vits14_pretrain.pth \
  --pretrained /home/notebook/data/group/slc/pixrestore/outputs/local_pd14_test-ppu4/checkpoints/checkpoint-00250000 \
  --resolution 512 \
  --global-batch-size 16 \
  --num-workers 16 \
  --max-steps 100000 \
  --checkpoint-steps 50000 \
  --eval-steps 10000 \
  "$@"
