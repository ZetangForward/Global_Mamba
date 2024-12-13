#!/bin/bash
#SBATCH --job-name=test    # 作业名称
#SBATCH --output=test.out  # 标准输出和标准错误的重定向路径
#SBATCH --error=test.err
#SBATCH --nodes=1            # 请求的节点数
#SBATCH --ntasks-per-node=1  # 每个节点的任务数
#SBATCH --cpus-per-task=52   # 每个任务的CPU数
#SBATCH --time=10000:00:00      # 执行时间 (HH:MM:SS)
#SBATCH --partition=hitsz_mzhang  # 分区

python /UNICOMFS/hitsz_khchen_4/zecheng/modelzipper/projects/state-space-model/scripts/sbatch/sleep.py
