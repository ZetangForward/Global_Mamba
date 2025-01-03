export CUDA_VISIBLE_DEVICES=4,5,6
LR=1e-3
mn=mamba-130m-hf
en=mamba-130m-v0-standard-$LR
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
    --lr_rate $LR \
    --train_strategy "ddp" \
    --monitor_metric "valid_lm_loss" \
    --train_batch_size 60 \
    --accumulate_grad_batches 4 \
    --val_batch_size 12 \
    --max_epochs 30 \
    --version $en \
    --data_dir MQAR/analysis/position-standard/v0-standard ;

# python src/test_dev2.py \
#     -mn $mn \
#     -en ${en} \
#     -pn amax_a100 \
#     -dn mqar \
#     --state eval \
#     --version $en \
#     --val_batch_size 1 \
#     --inference_mode \
#     --max_seq_len 16384 \
#     --nworkers 0 \
#     --ckpt_path ${model_path} \
#     --module_type attn \
#     --data_dir MQAR/analysis/position-standard/v0-standard;


# --module_tyep "longconv-gatedconv-decay-directdecay"
# --long_conv_kernel
# --decay-rate
# --mix_type