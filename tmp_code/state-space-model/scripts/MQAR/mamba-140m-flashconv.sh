export CUDA_VISIBLE_DEVICES=0,1
mn=mamba-130m-hf
en=mamba-130m-v0-standard-flashlongconv-128-on-standard
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
    --lr_rate 1e-3 \
    --train_strategy "ddp" \
    --monitor_metric "valid_lm_loss" \
    --train_batch_size 36 \
    --val_batch_size 12 \
    --max_epochs 30 \
    --version $en \
    --long_conv_kernel 128 \
    --module_type flashlongconv \
    --data_dir MQAR/analysis/position-standard/v0-standard;

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
#     --ckpt_path '/nvme1/zecheng/ckpt/mamba-130m-v0-standard-longconv-128/mamba-130m-v0-standard-longconv-128/checkpoints/last.ckpt' \
#     --module_type "longconv" \
#     --long_conv_kernel 128 \
#     --data_dir MQAR/analysis/position-last/v0-last;


# python src/test_dev2.py \
#     -mn $mn \
#     -en ${en}-analysis \
#     -pn amax_a100 \
#     -dn mqar \
#     --state eval \
#     --version $en \
#     --val_batch_size 1 \
#     --inference_mode \
#     --max_seq_len 16384 \
#     --nworkers 0 \
#     --ckpt_path '/nvme1/zecheng/ckpt/mamba-130m-v0-standard-longconv-128/mamba-130m-v0-standard-longconv-128/checkpoints/last.ckpt' \
#     --module_type "longconv" \
#     --long_conv_kernel 128 \
#     --record_debug_params \
#     --test_path MQAR/analysis/position-standard/v0-standard/test_for_analysis.jsonl;


# --module_tyep "longconv-gatedconv-decay-directdecay"
# --long_conv_kernel
# --decay-rate
# --mix_type