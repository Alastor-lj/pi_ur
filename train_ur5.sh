#!/bin/bash
# openpi JAX LoRA 微调: pi05_ur5 — 4卡 FSDP (GPU 4-7) + wandb
# 权重已缓存: ~/.cache/openpi/openpi-assets/checkpoints/pi05_base
set -e
cd /home/alastor/openpi/openpi_src
export PATH="$HOME/.local/bin:$PATH"
export HTTPS_PROXY=http://127.0.0.1:7893 HTTP_PROXY=http://127.0.0.1:7893
export HF_LEROBOT_HOME=/home/alastor/openpi/data
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
# NCCL 兼容性: 禁用 P2P/IB 走共享内存, 规避 PCIe 拓扑下的集合通信死锁
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

uv run scripts/train.py pi05_ur5 \
  --exp-name ur5_gamepad \
  --overwrite \
  --num-train-steps 20000 \
  --fsdp-devices 4 \
  --project-name pi05-ur5 \
  2>&1 | tee /home/alastor/openpi/outputs/train_ur5_openpi_jax.log
