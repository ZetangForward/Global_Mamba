export CUDA_VISIBLE_DEVICES=0,1,2,3

# CUDA_LAUNCH_BLOCKING=1 torchrun --nnode=1 --nproc_per_node=1 --master_port 3912 src/train_dev2.py \

torchrun --nnode=1 --nproc_per_node=8 src/train_dev2.py \
    -mn mamba-370m-km \
    -pn hitsz \
    -en slimpajama \
    -dn slimpajama \
    --node_num 1 \
    --device_num 2 \
    --state train \
    --train_batch_size 6 \
    --monitor_metric "train_lm_loss" \
    --save_top_k 5 \
    --every_n_train_steps 400 \
    --accumulate_grad_batches 2 \
    --val_batch_size 1 \
    --input_seq_len 2000 \
    --max_training_steps 5000 \
    --use_deepspeed;


