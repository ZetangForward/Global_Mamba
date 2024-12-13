export CUDA_VISIBLE_DEVICES=7
mn=mamba-1_4b-hf
en=mamba-1_4b-hf-k4v8
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
#     --lr_rate 3e-3 \
#     --train_strategy "ddp" \
#     --monitor_metric "valid_lm_loss" \
#     --train_batch_size 4 \
#     --val_batch_size 2 \
#     --max_epochs 30 \
#     --version $en \
#     --data_dir MQAR/mqar-v6-k4v8-512-seq_loss ;


python src/test_dev2.py \
    -mn $mn \
    -en ${en}_1 \
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
    
