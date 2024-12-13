export CUDA_VISIBLE_DEVICES=0,1,2,3,4,7

torchrun --nnode=1 --nproc_per_node=6 src/train_dev2.py \
    -mn mamba-370m-km \
    --ckpt_path /nvme1/zecheng/ckpt/mamba-370m-km2/version_1/checkpoints/last.ckpt/model.bin \
    -pn amax_a100 \
    -en mamba-370m-km2 \
    -dn longalpaca \
    --node_num 1 \
    --state train \
    --train_batch_size 1 \
    --accumulate_grad_batches 16 \
    --val_batch_size 1 \
    --max_epochs 10 \
    --max_seq_len 8000 \
    --monitor_metric "train_lm_loss" \
    --version alpaca;