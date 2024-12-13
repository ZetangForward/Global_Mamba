export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nnode=1 --nproc_per_node=4 src/train_dev2.py \
    -mn tiny_mamba-k4 \
    -pn amax_a100 \
    -en mqar-tiny_mamba-k4 \
    -dn mqar \
    --node_num 1 \
    --state train \
    --train_batch_size 1 \
    --accumulate_grad_batches 24 \
    --val_batch_size 1 \
    --max_epochs 10 \
    --monitor_metric "train_lm_loss" \
    --version longalpaca_fsdp \
    --train_stratefy 'fsdp' \
    --model_module models.custom_mamba_v3 \
    --block_name MambaBlock \



