#!/bin/bash

# 设置环境变量
export PYTHONUNBUFFERED="1"
export MASTER_PORT="23456"
export PET_MASTER_PORT="23456"
export MASTER_ADDR="127.0.0.1"
export PET_MASTER_ADDR="127.0.0.1"
export WORLD_SIZE="1"
export RANK="0"
export PET_NPROC_PER_NODE="auto"
export PET_NODE_RANK="0"
export PET_NNODES="1"

# 定义 Python 脚本的路径，假设它位于当前目录下的 project/src/pytorch_ddp_example/__main__.py
script_path="./src/pytorch_ddp_example/__main__.py"

# 检查 Python 脚本是否存在
if [ ! -f "$script_path" ]; then
  echo "Error: Python script not found at $script_path"
  exit 1
fi

# 执行 Python 脚本
python $script_path --epochs=9 \
                    --batch-size=64 \
                    --test-batch-size=1000 \
                    --lr=0.01 \
                    --momentum=0.5 \
                    --seed=1 \
                    --log-interval=10 \
                    --dir=logs \
                    --backend=nccl \
                    --ckpt-path="checkpoint/model_checkpoint"
                    # --dataset-mirror=http://registry.cn:9000/dataset/