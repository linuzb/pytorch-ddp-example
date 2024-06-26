#!/bin/bash

# 检查是否提供了image参数
if [ -z "$1" ]; then
  echo "未提供image参数，使用默认值"
  image="registry.cn-hangzhou.aliyuncs.com/linuzb/pytorch:pytorch-ddp-01fcd18"
else
  image=$1
fi

echo "使用的image是: $image"

docker run --gpus all \
  -e PYTHONUNBUFFERED="1" \
  -e MASTER_PORT="23456" \
  -e PET_MASTER_PORT="23456" \
  -e MASTER_ADDR="127.0.0.1" \
  -e PET_MASTER_ADDR="127.0.0.1" \
  -e WORLD_SIZE="1" \
  -e RANK="0" \
  -e PET_NPROC_PER_NODE="auto" \
  -e PET_NODE_RANK="0" \
  -e PET_NNODES="1" \
  -p 23456:23456 \
  ${image} \
  --epochs=50 \
  --batch-size=128 \
  --test-batch-size=1000 \
  --lr=0.01 \
  --momentum=0.5 \
  --seed=1 \
  --log-interval=10 \
  --dir=logs \
  --backend=nccl