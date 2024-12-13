export CUDA_VISIBLE_DEVICES=0
mn=rwkv
en=rwkv-v0-k4v8
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
#     --lr_rate 1e-3 \
#     --train_strategy "ddp" \
#     --monitor_metric "valid_lm_loss" \
#     --train_batch_size 48 \
#     --val_batch_size 32 \
#     --max_epochs 30 \
#     --version $en \
#     --hf_trainer \
#     --data_dir /nvme1/zecheng/data/MQAR/mqar-v6-k4v8-512-seq_loss ;

python src/test_dev2.py \
    -mn $mn \
    -en ${en}-k4v8 \
    -pn amax_a100 \
    -dn mqar \
    --state eval \
    --version $en \
    --val_batch_size 1 \
    --inference_mode \
    --max_seq_len 16384 \
    --nworkers 0 \
    --ckpt_path ${model_path} \
    --data_dir /nvme1/zecheng/data/MQAR/mqar-v6-k4v8-512-seq_loss;