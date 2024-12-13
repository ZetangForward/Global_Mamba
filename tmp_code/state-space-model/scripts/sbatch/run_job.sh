#!/bin/bash
#SBATCH --job-name=zecheng_1    # 作业名称
#SBATCH --output=logs.out  # 标准输出和标准错误的重定向路径
#SBATCH --error=logs.err
#SBATCH --ntasks-per-node=1  # 每个节点的任务数
#SBATCH --cpus-per-task=48   # 每个任务的CPU数
#SBATCH --time=10000:00:00   # 执行时间 (HH:MM:SS)
#SBATCH --partition=hitsz_mzhang  # 分区

# 加载模块
module load anaconda/2023.03 
module load nccl/2.18.1-cuda-11.8 
module load openmpi/4.1.5-gcc-9.4.0

# 激活conda环境
source activate
conda activate zecheng

# 进入你的工作目录
cd /UNICOMFS/hitsz_khchen_4/zecheng/modelzipper/projects/state-space-model

# 执行你的 bash 脚本
bash scripts/pretrain/slimpajama-mamba-km.sh
