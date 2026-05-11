#!/bin/bash
#SBATCH -p compute --exclusive
#SBATCH -N 1
#SBATCH -t 1-00:00:00			
#SBATCH --ntasks-per-node=1           
#SBATCH -A lt200473
#SBATCH -J dataset
#SBATCH --output=tokenized.out
                  
module load Mamba
conda deactivate
conda activate /project/lt200473-ttctvs/workshop-pretrain/env
export HF_DATASETS_CACHE="/scratch/lt200473-ttctvs/.cache"
export HF_HOME="/scratch/lt200473-ttctvs/.cache"
export HF_HUB_CACHE="/scratch/lt200473-ttctvs/.cache"
export HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1

python generate_datasets.py \
    --input_dir "/project/lt200473-ttctvs/workshop-pretrain/data/json/*" \
    --model_name "/project/lt200473-ttctvs/workshop-pretrain/model/Qwen3-0.6B-Base" \
    --output_path "./outputs/tokenized-data" \
    --context_length 8192 \
    --num_proc 32 \
    --batch_size 5000 \
    --num_chunks 5



