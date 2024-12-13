export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nnode=1 --nproc_per_node=4 src/train_dev2.py \
    -mn mamba-370m-hf \
    -pn hitsz \
    -en longalpaca-mamba-370m-hf \
    -dn longalpaca \
    --node_num 1 \
    --state train \
    --train_batch_size 2 \
    --accumulate_grad_batches 6 \
    --val_batch_size 1 \
    --max_epochs 10 \
    --max_seq_len 6000 \
    --monitor_metric "train_lm_loss" \
    --version 1;


