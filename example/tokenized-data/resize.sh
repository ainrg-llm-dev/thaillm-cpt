#!/bin/bash
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 1-00:00:00			
#SBATCH --ntasks-per-node=1           
#SBATCH -A lt200258
#SBATCH -J resize
#SBATCH --output=resize.out

module load Mamba
conda deactivate
conda activate ./../../env
export CATCH_DIR="./../../.cache"
export HF_DATASETS_CACHE=CATCH_DIR
export HF_HOME=CATCH_DIR
export HF_HUB_CACHE=CATCH_DIR
export HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1

python resize_context.py \
    --input_path "./outputs/tokenized-data-wikipedia" \
    --output_path "./outputs/tokenized-data-wikipedia-4096" \
    --new_context_length 4096 \
    --batch_size 3000 \
    --num_proc 64



