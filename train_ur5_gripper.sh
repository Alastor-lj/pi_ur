#!/bin/bash
# openpi JAX LoRA: pi05_ur5_gripper — 单卡 (规避多卡 NCCL 静默损坏, openpi issue #822)
# 单卡+batch32 为已验证健康配方; 每1000步保存; 保存竞态已修复
set -e
cd /home/alastor/openpi/openpi_src
export PATH="$HOME/.local/bin:$PATH"
export HTTPS_PROXY=http://127.0.0.1:7893 HTTP_PROXY=http://127.0.0.1:7893
export HF_LEROBOT_HOME=/home/alastor/openpi/data
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=4
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

uv run scripts/train.py pi05_ur5_gripper \
  --exp-name gripper_v4_singlegpu \
  --overwrite \
  --project-name pi05-ur5 \
  2>&1 | tee /home/alastor/openpi/outputs/train_ur5_gripper_jax.log
