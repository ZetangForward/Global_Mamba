export CUDA_VISIBLE_DEVICES=6
mn=mamba-130m-hf
en=mamba-130m-hf-v0-last-robustness
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
random_port=$(( (RANDOM % 10000) + 1024 ))
ckpt_dir=/nvme1/zecheng/ckpt/$en/$en/checkpoints
model_path=${ckpt_dir}/last.ckpt
# torchrun --nnode=1 --nproc_per_node=${num_devices} --master_port ${random_port} src/train_dev2.py \
#     -mn $mn \
#     -pn amax_a100 \
#     -en $en\
#     -dn mqar \
#     --node_num 1 \
#     --device_num ${num_devices} \
#     --nworkers 4 \
#     --state train \
#     --train_strategy "ddp" \
#     --monitor_metric "valid_lm_loss" \
#     --train_batch_size 44 \
#     --val_batch_size 32 \
#     --max_epochs 30 \
#     --version $en \
#     --delta_ratio 0 \
#     --data_dir MQAR/analysis/position-standard/v6-standard;

# wait 

python src/test_dev2.py \
    -mn $mn \
    -en ${en}_test_on_robustness_split \
    -pn amax_a100 \
    -dn mqar \
    --state eval \
    --version $en \
    --val_batch_size 1 \
    --inference_mode \
    --max_seq_len 16384 \
    --nworkers 0 \
    --delta_ratio 0 \
    --ckpt_path /nvme1/zecheng/ckpt/mamba-130m-hf-v0-position-last/mamba-130m-hf-v0-position-last/checkpoints/last.ckpt \
    --data_dir MQAR/analysis/robustness_v0_split4;
    

