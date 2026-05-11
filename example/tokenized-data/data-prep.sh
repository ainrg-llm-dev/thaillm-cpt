#!/bin/bash
#SBATCH -p compute-devel --exclusive
#SBATCH -N 1
#SBATCH -t 2:00:00			
#SBATCH --ntasks-per-node=1           
#SBATCH -A lt200258
#SBATCH -J dataset
#SBATCH --output=tokenized.out
                  
module load Mamba
conda deactivate
conda activate ./../../env
export CATCH_DIR="./../../.cache"
export HF_DATASETS_CACHE=$CATCH_DIR
export HF_HOME=$CATCH_DIR
export HF_HUB_CACHE=$CATCH_DIR
export HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1

python generate_datasets.py \
    --input_dir "./../../data/wikipedia-th/20231101.th/*" \
    --model_name "Qwen/Qwen3-0.6B-Base" \
    --output_path "./outputs/tokenized-data-wikipedia" \
    --context_length 8192 \
    --num_proc 32 \
    --batch_size 5000 \
    --num_chunks 5



