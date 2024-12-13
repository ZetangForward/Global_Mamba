export CUDA_VISIBLE_DEVICES=0
mn=gla-130m
en=gla-130m-v0-standard-1e-2
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
random_port=$(( (RANDOM % 10000) + 1024 ))
ckpt_dir=/nvme1/zecheng/ckpt/$en/$en/checkpoints
model_path=${ckpt_dir}/last.ckpt
torchrun --nnode=1 --nproc_per_node=${num_devices} --master_port ${random_port} src/train_dev2.py \
    -mn $mn \
    -pn amax_a100 \
    -en $en\
    -dn mqar \
    --node_num 1 \
    --device_num ${num_devices} \
    --nworkers 0 \
    --state train \
    --lr_rate 1e-2 \
    --train_strategy "ddp" \
    --monitor_metric "valid_lm_loss" \
    --train_batch_size 192 \
    --val_batch_size 32 \
    --max_epochs 30 \
    --version $en \
    --hf_trainer \
    --data_dir MQAR/analysis/position-standard/v0-standard ;



# export CUDA_VISIBLE_DEVICES=4,5,6,7
# mn=gla-130m
# en=gla-130m-v0-shuffle-1e-2
# num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
# random_port=$(( (RANDOM % 10000) + 1024 ))
# ckpt_dir=/nvme1/zecheng/ckpt/$en/$en/checkpoints
# model_path=${ckpt_dir}/last.ckpt
# torchrun --nnode=1 --nproc_per_node=${num_devices} --master_port ${random_port} src/train_dev2.py \
#     -mn $mn \
#     -pn amax_a100 \
#     -en $en\
#     -dn mqar \
#     --node_num 1 \
#     --device_num ${num_devices} \
#     --nworkers 0 \
#     --state train \
#     --lr_rate 1e-2 \
#     --train_strategy "ddp" \
#     --monitor_metric "valid_lm_loss" \
#     --train_batch_size 132 \
#     --val_batch_size 32 \
#     --max_epochs 30 \
#     --version $en \
#     --hf_trainer \
#     --data_dir MQAR/analysis/position-shuffle/v0-shuffle ;

# python src/test_dev2.py \
#     -mn $mn \
#     -en ${en}-onstandard \
#     -pn amax_a100 \
#     -dn mqar \
#     --state eval \
#     --version $en \
#     --val_batch_size 1 \
#     --inference_mode \
#     --max_seq_len 16384 \
#     --nworkers 0 \
#     --ckpt_path ${model_path} \
#     --data_dir MQAR/analysis/position-standard/v0-standard;