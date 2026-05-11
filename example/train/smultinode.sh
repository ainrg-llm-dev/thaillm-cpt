#!/bin/bash

(
  sleep 3600
  echo "==== NVIDIA-SMI after 1 hour ===="
  nvidia-smi
) &

# Main training
echo "Starting Llama-factory training..."
echo "Number of nodes: $SLURM_NNODES"
echo "Node ID: $SLURM_PROCID"
echo "Master IP: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"

echo "Starting training..."
 
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES" 

export WORLD_SIZE=$((SLURM_NNODES * $GPUS_PER_NODE))
echo "WORLD_SIZE set to: $WORLD_SIZE"
export TRITON_CACHE_DIR="/scratch/lt200473-ttctvs/cache/triton_cache_$SLURM_PROCID"

FORCE_TORCHRUN=1 NNODES=$SLURM_NNODES NODE_RANK=$SLURM_PROCID MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT llamafactory-cli train train.yaml
