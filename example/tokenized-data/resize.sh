#!/bin/bash
#SBATCH -p compute --exclusive
#SBATCH -N 1
#SBATCH -t 1-00:00:00			
#SBATCH --ntasks-per-node=1           
#SBATCH -A lt200473
#SBATCH -J resize
#SBATCH --output=resize.out

module load Mamba
conda deactivate
conda activate /project/lt200473-ttctvs/workshop-pretrain/env
export HF_DATASETS_CACHE="/scratch/lt200473-ttctvs/.cache"
export HF_HOME="/scratch/lt200473-ttctvs/.cache"
export HF_HUB_CACHE="/scratch/lt200473-ttctvs/.cache"
export HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1

python resize_context.py \
    --input_path "./outputs/tokenized-data" \
    --output_path "./outputs/tokenized-data-4096" \
    --new_context_length 4096 \
    --batch_size 3000 \
    --num_proc 64



