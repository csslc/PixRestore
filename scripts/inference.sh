#!/usr/bin/env bash
set -euo pipefail

cd /home/notebook/data/group/slc/pixrestore

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch \
  --multi_gpu \
  --num_processes 8 \
  --main_process_port 29619 \
  inference.py \
  --config configs/train.yaml \
  -c /home/notebook/data/group/slc/pixrestore/outputs/local_pd_gan_test-250k-ppu5/checkpoints/checkpoint-00100000 \
  -i /home/notebook/data/group/slc/Datasets/IR_testdata/PixelRestore/synthetic_testdata.json \
  -o /home/notebook/data/group/slc/Datasets/IR_testdata/PixelRestore/results \
  --test-mode center_crop \
  --infer-steps 1 \
  --cfg-scale 1.0 \
  --cond-strength-aelq-test 1.0 \
  --weak-cond-strength-aelq 1.0 \
  --method pixrestore-s-retrain-250k \
  --seed 0
