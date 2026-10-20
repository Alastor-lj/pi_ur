#!/bin/bash
# openpi JAX LoRA: pi05_ur5_15hz — DROID 模板对齐版 (单卡, 已验证稳定路径)
# 数据: ur5_openpi_15hz_v21 (50 eps, 15Hz, 7维含夹爪, pick-and-place)
# 关键: H=16@15Hz=1.07s时窗 / discrete_state_input=True / 归一化后钳位±3
set -e
cd /home/alastor/openpi/openpi_src
export PATH="$HOME/.local/bin:$PATH"
export HTTPS_PROXY=http://127.0.0.1:7893 HTTP_PROXY=http://127.0.0.1:7893
export HF_LEROBOT_HOME=/home/alastor/openpi/data
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=4
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

uv run scripts/train.py pi05_ur5_15hz \
  --exp-name ur5_15hz_v1 \
  --overwrite \
  --project-name pi05-ur5 \
  2>&1 | tee /home/alastor/openpi/outputs/train_ur5_15hz.log
