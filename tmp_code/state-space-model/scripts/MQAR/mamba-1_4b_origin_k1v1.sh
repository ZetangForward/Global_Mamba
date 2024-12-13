export CUDA_VISIBLE_DEVICES=4,5,6,7
mn=mamba-1_4b-hf
en=mamba-1_4b-hf-position-standard-1e-4
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
#     --nworkers 0 \
#     --state train \
#     --lr_rate 1e-4 \
#     --train_strategy "ddp" \
#     --monitor_metric "valid_lm_loss" \
#     --train_batch_size 16 \
#     --val_batch_size 5 \
#     --max_epochs 30 \
#     --version $en \
#     --ckpt_path /nvme1/zecheng/ckpt/mamba-1_4b-hf-position-standard-3e-3/mamba-1_4b-hf-position-standard-3e-3/checkpoints/epoch=2-step=12186-valid_lm_loss=0.04.ckpt \
#     --data_dir /nvme1/zecheng/data/MQAR/analysis/position-standard/v0-standard ;


python src/test_dev2.py \
    -mn $mn \
    -en ${en}_on_shuffle \
    -pn amax_a100 \
    -dn mqar \
    --state eval \
    --version $en \
    --val_batch_size 16 \
    --inference_mode \
    --max_seq_len 16384 \
    --nworkers 0 \
    --ckpt_path ${model_path} \
    --data_dir /nvme1/zecheng/data/MQAR/analysis/position-shuffle/v0-shuffle;
    
