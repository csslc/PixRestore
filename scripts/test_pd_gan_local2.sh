#!/usr/bin/env bash
# Continue local_pd14_test-ppu4 (DiT-S + DINOv2-S) with DINO-GAN finetuning.
# Architecture is pinned in configs/train_gan.yaml so it does not inherit DiT-B.
set -euo pipefail

cd /home/notebook/data/group/slc/pixrestore

SOURCE_ROOT=/home/notebook/data/group/slc/pisa-sr

MANIFESTS=(
  "$SOURCE_ROOT/dataloaders/slc_5_3_uav_sync/all.json"
  "$SOURCE_ROOT/dataloaders/slc_7_10_sep_all_data/df2k_gaussian_noise.json"
  "$SOURCE_ROOT/dataloaders/slc_7_10_sep_all_data/gopro.json"
  "$SOURCE_ROOT/dataloaders/slc_7_10_sep_all_data/loldataset.json"
  "$SOURCE_ROOT/dataloaders/slc_7_10_sep_all_data/lsd_defocus.json"
  "$SOURCE_ROOT/dataloaders/slc_7_10_sep_all_data/polyu.json"
  "$SOURCE_ROOT/dataloaders/slc_7_10_sep_all_data/rainds_real_raindrop_streak.json"
  "$SOURCE_ROOT/dataloaders/slc_7_10_sep_all_data/rainds_real.json"
  "$SOURCE_ROOT/dataloaders/slc_7_10_sep_all_data/realrain_1k.json"
  "$SOURCE_ROOT/dataloaders/slc_7_10_sep_all_data/realsr.json"
  "$SOURCE_ROOT/dataloaders/slc_7_10_sep_all_data/reside_6k.json"
  "$SOURCE_ROOT/dataloaders/slc_7_10_sep_all_data/screensr.json"
  "$SOURCE_ROOT/dataloaders/slc_7_10_sep_all_data/uav_rain1k.json"
  "$SOURCE_ROOT/dataloaders/slc_7_10_sep_all_data/uhd_blur.json"
  "$SOURCE_ROOT/dataloaders/slc_7_10_sep_all_data/uhd_haze.json"
  "$SOURCE_ROOT/dataloaders/slc_7_10_sep_all_data/uhd_ll.json"
  "$SOURCE_ROOT/dataloaders/slc_7_10_sep_all_data/weatherbench.json"
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
  --output-dir outputs/local_pd_gan_test-samedata-ppu4 \
  --dinov2-checkpoint /home/notebook/data/group/slc/pisa-sr/pretrain_models/pisasr_preview/torch_cache/checkpoints/dinov2_vits14_pretrain.pth \
  --pretrained /home/notebook/data/group/slc/pixrestore/outputs/local_pd14_test-ppu4/checkpoints/checkpoint-00400000 \
  --resolution 512 \
  --global-batch-size 16 \
  --num-workers 16 \
  --max-steps 100000 \
  --checkpoint-steps 50000 \
  --eval-steps 10000 \
  "$@"
