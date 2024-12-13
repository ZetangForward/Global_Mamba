Debug Config

module load anaconda/2023.03 
module load nccl/2.18.1-cuda-11.8 
module load openmpi/4.1.5-gcc-9.4.0
source activate
conda activate zecheng

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "mamba_debug_train",
            "type": "python",
            "request": "launch",
            "program": "/nvme/zecheng/modelzipper/projects/state-space-model/src/train_dev2.py",
            "console": "integratedTerminal",
            "cwd": "/nvme/zecheng/modelzipper/projects/state-space-model",
            "env": {
                "CUDA_VISIBLE_DEVICES": "7"
            },
            "python": "/home/amax/anaconda3/envs/zecheng/bin/python",
            "justMyCode": false,
            "args": [
                "--model_name_or_path", "mamba-370m-k8",
                "--platform_name", "amax_a100",
                "--experiment_name", "test",
                "--ckpt_path", "/nvme/zecheng/ckpt/h_800/ckpt/slimpajama/mamba_370m_big_kernel-k8/checkpoints/last.ckpt/model.bin"
            ]
        },
        {
            "name": "mamba_analysis",
            "type": "python",
            "request": "launch",
            "program": "/nvme/zecheng/modelzipper/projects/state-space-model/src/analysis.py",
            "console": "integratedTerminal",
            "cwd": "/nvme/zecheng/modelzipper/projects/state-space-model",
            "env": {
                "CUDA_VISIBLE_DEVICES": "7"
            },
            "python": "/home/amax/anaconda3/envs/zecheng/bin/python",
            "justMyCode": false,
        },
        {
            "name": "mamba_debug_train",
            "type": "python",
            "request": "launch",
            "program": "/nvme/zecheng/modelzipper/projects/state-space-model/src/train.py",
            "console": "integratedTerminal",
            "cwd": "/nvme/zecheng/modelzipper/projects/state-space-model",
            "env": {
                "CUDA_VISIBLE_DEVICES": "7"
            },
            "python": "/home/amax/anaconda3/envs/zecheng/bin/python",
            "justMyCode": false,
        },
        {
            "name": "mamba_analysis",
            "type": "python",
            "request": "launch",
            "program": "/nvme/zecheng/modelzipper/projects/state-space-model/src/analysis.py",
            "console": "integratedTerminal",
            "cwd": "/nvme/zecheng/modelzipper/projects/state-space-model",
            "env": {
                "CUDA_VISIBLE_DEVICES": "7"
            },
            "python": "/home/amax/anaconda3/envs/zecheng/bin/python",
            "justMyCode": false,
        },
        {
            "name": "mamba_debug_train",
            "type": "python",
            "request": "launch",
            "program": "/nvme/zecheng/modelzipper/projects/state-space-model/src/train.py",
            "console": "integratedTerminal",
            "cwd": "/nvme/zecheng/modelzipper/projects/state-space-model",
            "env": {
                "CUDA_VISIBLE_DEVICES": "7"
            },
            "python": "/home/amax/anaconda3/envs/zecheng/bin/python",
            "justMyCode": false,
        },
    ]
}

```


export CUDA_VISIBLE_DEVICES=7

# CUDA_LAUNCH_BLOCKING=1 torchrun --nnode=1 --nproc_per_node=1 --master_port 3912 src/train_dev2.py \

python src/train_dev2.py \
    -mn mamba-370m-km \
    -pn langchao \
    -en longalpaca_test \
    -dn longalpaca \
    --node_num 1 \
    --device_num 1 \
    --state train \
    --train_batch_size 4 \
    --monitor_metric "train_lm_loss" \
    --save_top_k 5 \
    --every_n_train_steps 2000 \
    --accumulate_grad_batches 4 \
    --val_batch_size 1 \
    --max_epochs 50 \
    --debug;


