#!/bin/bash
#SBATCH -p gpu  # Specify partition [compute/memory/gpu]
#SBATCH -N 1                 # จำนวนโหนดที่ต้องการใช้
#SBATCH -c 16                 # Specify number of CPU cores
#SBATCH --gpus-per-task=4           # จำนวน GPU ที่ต้องการต่อ 1 node ในกรณีที่ใช้ compute node ให้ตั้งเป็น 0
#SBATCH --ntasks-per-node=1         # Specify tasks per node (ตรงนี้อย่าไปแก้)
#SBATCH -t 24:00:00                  # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt200473                 # บอกว่าคิดตังได้ที่ project ไหน sbalance -g เช็คดูว่ามีเงินจาก project ไหนบ้าง
#SBATCH -J train        # ตั้งชื่อ job ให้หาอ่านง่าย ๆ
#SBATCH -o ./logs/%j/%j.out        # ตั้งชื่อไฟล์ output
#SBATCH --reservation=thaisc_311

current_date_time="`date "+%Y-%m-%d %H:%M:%S"`";
echo $current_date_time;

export HF_HOME=/scratch/lt200473-ttctvs/.cache
export HF_HUB_CACHE=/scratch/lt200473-ttctvs/.cache
export HF_DATASETS_CACHE=/scratch/lt200473-ttctvs/.cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=2
export NCCL_TIMEOUT=360000
export CUDA_LAUNCH_BLOCKING=1
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_EXTENSIONS_DIR=/scratch/lt200473-ttctvs/.cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:40960
export WANDB_MODE="offline"

######################
### load Module ###
######################
module restore
module load Mamba
# module load PrgEnv-gnu
module load cpe-cuda
# 
module load gcc 
module load cuda
echo "CUDA HOME: $CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

######################
### Set enviroment ###
######################
conda deactivate
conda activate /project/lt200473-ttctvs/workshop-pretrain/env

which python

######################
#### Set network #####
######################
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802 #12999 #12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo "Number of nodes: $COUNT_NODE"
######################

######################
### Set OMP_NUM_THREADS FOR LLAMA_FACTORY CPU ###
######################
export OMP_NUM_THREADS=1
export DS_SKIP_CUDA_CHECK=1
export DS_BUILD_AIO=1
export DS_BUILD_CPU_ADAM=1
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=0               # Enable InfiniBand
export NCCL_NET_GDR_LEVEL=5            # GPU Direct RDMA
export DISABLE_VERSION_CHECK=1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
LOG_DIR="./logs/$SLURM_JOB_ID"
mkdir -p $LOG_DIR/
export LOG_DIR=$LOG_DIR

### ปกติเนื่องจาก script นี้เอาไว้ใช้รัน multi-node + multi-gpu แต่ด้วยทรัพยากรณ์ที่มีอยู่ตอนนี้เลยตั้งเป็น 1
export GPUS_PER_NODE=1 # ถ้ามี 4 ตัว/node ก็ใส่เป็น 4 
export CUDA_VISIBLE_DEVICES="0" # ถ้ามี 4 ตัว ก็ใส่เป็น "0,1,2,3"

srun --output=${LOG_DIR}/node_log/node-%t.out sh smultinode.sh


