#!/bin/bash
#SBATCH -p gpu-devel  # Specify partition [compute/memory/gpu]
#SBATCH -N 1                 # จำนวนโหนดที่ต้องการใช้
#SBATCH -c 16                 # Specify number of CPU cores
#SBATCH --gpus-per-task=4           # จำนวน GPU ที่ต้องการต่อ 1 node ในกรณีที่ใช้ compute node ให้ตั้งเป็น 0
#SBATCH --ntasks-per-node=1         # Specify tasks per node (ตรงนี้อย่าไปแก้)
#SBATCH -t 2:00:00                  # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt200258                 # บอกว่าคิดตังได้ที่ project ไหน sbalance -g เช็คดูว่ามีเงินจาก project ไหนบ้าง
#SBATCH -J train        # ตั้งชื่อ job ให้หาอ่านง่าย ๆ
#SBATCH -o ./logs/%j/%j.out        # ตั้งชื่อไฟล์ output

current_date_time="`date "+%Y-%m-%d %H:%M:%S"`";
echo $current_date_time;
export CATCH_DIR="./../../.cache" # recommend to set to a local directory with enough space, e.g. /scratch/$USER/.cache
export HF_HOME=$CATCH_DIR
export HF_HUB_CACHE=$CATCH_DIR
export HF_DATASETS_CACHE=$CATCH_DIR
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=2
export NCCL_TIMEOUT=360000
export CUDA_LAUNCH_BLOCKING=1
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_EXTENSIONS_DIR=$CATCH_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE="offline"

######################
### load Module ###
######################
module restore
module load Mamba
module load cpe-cuda
module load gcc 
module load cuda
echo "CUDA HOME: $CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

######################
### Set enviroment ###
######################
conda deactivate
conda activate ./../../env

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

### ปกติเนื่องจาก script นี้เอาไว้ใช้รัน multi-node + multi-gpu แต่หากใครมีแค่ 1 node + 1 gpu ก็สามารถรันได้โดยไม่ต้องแก้ไขอะไรเลย แต่ถ้ามีหลาย gpu ต่อ node ก็ให้แก้ไขตามที่ comment ไว้ด้านล่างนี้ครับ
export GPUS_PER_NODE=4 # ถ้ามี 1 ตัว/node ก็ใส่เป็น 1 
export CUDA_VISIBLE_DEVICES="0,1,2,3" # ถ้ามี 1 ตัว ก็ใส่เป็น 0

srun --output=${LOG_DIR}/node_log/node-%t.out sh smultinode.sh


