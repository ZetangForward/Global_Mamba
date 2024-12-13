export CUDA_VISIBLE_DEVICES=7
mn=mamba-tiny
en=mamba-tiny-position-standard-3e-3
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
#     --train_batch_size 120 \
#     --val_batch_size 20 \
#     --max_epochs 15 \
#     --ckpt_path /nvme1/zecheng/ckpt/mamba-tiny-position-standard-longconv-128-3e-3/version_1/checkpoints/last.ckpt \
#     --data_dir /nvme1/zecheng/data/MQAR/analysis/position-standard/v0-standard;

python src/test_dev2.py \
    -mn $mn \
    -en ${en}-on_shuffle \
    -pn amax_a100 \
    -dn mqar \
    --state eval \
    --version $en \
    --val_batch_size 1 \
    --inference_mode \
    --nworkers 0 \
    --ckpt_path /nvme1/zecheng/ckpt/mamba-tiny-k1v1-standard/version_1/checkpoints/last.ckpt\
    --data_dir /nvme1/zecheng/data/MQAR/analysis/position-shuffle/v0-shuffle;



# python src/test_dev2.py \
#     -mn $mn \
#     -en ${en}-robustness \
#     -pn amax_a100 \
#     -dn mqar \
#     --state eval \
#     --version $en \
#     --val_batch_size 1 \
#     --inference_mode \
#     --max_seq_len 16384 \
#     --nworkers 0 \
#     --ckpt_path ${model_path} \
#     --module_type "longconv" \
#     --long_conv_kernel 128 \
#     --test_path /nvme1/zecheng/data/MQAR/analysis/mqar-v6-k2v2-standard-robustness/test.jsonl;

    # --record_debug_params \

# --module_tyep "longconv-gatedconv-decay-directdecay"
# --long_conv_kernel
# --decay-rate
# --mix_type