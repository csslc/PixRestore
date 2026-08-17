#!/usr/bin/env bash
# Run a short local PD14 training test.
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
accelerate launch \
  --multi_gpu \
  --num_processes 8 \
  --main_process_port 29618 \
  train.py \
  --config configs/train.yaml \
  --manifest "${MANIFESTS[@]}" \
  --test-lq-dir /home/notebook/data/group/slc/Datasets/IR_testdata/selected_testsets/test_a_little/LQ \
  --test-gt-dir /home/notebook/data/group/slc/Datasets/IR_testdata/selected_testsets/test_a_little/GT \
  --output-dir outputs/local_pd14_test-dit-b-a8003 \
  --dinov2-checkpoint /home/notebook/data/group/slc/dinov2/pretrained_model/dinov2_vitb14_pretrain.pth \
  --global-batch-size 16 \
  --num-workers 16 \
  --max-steps 400000 \
  --checkpoint-steps 50000 \
  --eval-steps 10000
