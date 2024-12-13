export CUDA_VISIBLE_DEVICES=0,1,2,3,4,7
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

torchrun --nnode=1 --nproc_per_node=$num_devices src/train_dev2.py \
    -mn mamba-370m-km2 \
    -pn amax_a100 \
    -en mamba-370m-km2 \
    -dn slimpajama \
    --version improve_2 \
    --node_num 1 \
    --device_num $num_devices \
    --state train \
    --train_batch_size 4 \
    --max_epochs 50 \
    --monitor_metric "train_lm_loss" \
    --save_top_k 2 \
    --every_n_train_steps 300 \
    --accumulate_grad_batches 4 \
    --val_batch_size 4 \
    --max_seq_len 2048 \
    --max_training_steps 15000 \
    --ckpt_path "/nvme/hf_models/mamba-370m-hf/pytorch_model.bin" \
    --use_deepspeed;


