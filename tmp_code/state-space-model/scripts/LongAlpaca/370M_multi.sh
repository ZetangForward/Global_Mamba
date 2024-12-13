export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --nnode=1 --nproc_per_node=4 src/train_dev2.py \
    -mn mamba-370m-km \
    --ckpt_path /UNICOMFS/hitsz_khchen_4/zecheng/ckpt/longalpaca/version_1/checkpoints/last.ckpt \
    -pn hitsz \
    -en longalpaca \
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