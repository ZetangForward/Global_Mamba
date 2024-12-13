export CUDA_VISIBLE_DEVICES=7
# learning_rates=("3e-3")
# for lr in "${learning_rates[@]}"; do
#     mn=mamba-tiny
#     en=mamba-tiny-position-standard-longconv-128-$lr-post
#     num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
#     random_port=$(( (RANDOM % 10000) + 1024 ))
#     ckpt_dir=/nvme1/zecheng/ckpt/$en/$en/checkpoints
#     model_path=${ckpt_dir}/last.ckpt
#     # torchrun --nnode=1 --nproc_per_node=${num_devices} --master_port ${random_port} src/train_dev2.py \
#     #     -mn $mn \
#     #     -pn amax_a100 \
#     #     -en $en\
#     #     -dn mqar \
#     #     --node_num 1 \
#     #     --device_num ${num_devices} \
#     #     --nworkers 0 \
#     #     --state train \
#     #     --lr_rate $lr \
#     #     --train_strategy "ddp" \
#     #     --monitor_metric "valid_lm_loss" \
#     #     --train_batch_size 120 \
#     #     --val_batch_size 64 \
#     #     --max_epochs 15 \
#     #     --module_type "longconv" \
#     #     --long_conv_kernel 128 \
#     #     --ckpt_path /nvme1/zecheng/ckpt/mamba-tiny-position-standard-longconv-128-3e-3/version_1/checkpoints/last.ckpt \
#     #     --data_dir /nvme1/zecheng/data/MQAR/analysis/position-standard/v0-standard;

#     # wait 

#     mn=mamba-tiny
#     en=mamba-tiny-k2v2-longconv-128-$lr
#     num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
#     random_port=$(( (RANDOM % 10000) + 1024 ))
#     ckpt_dir=/nvme1/zecheng/ckpt/$en/$en/checkpoints
#     model_path=${ckpt_dir}/last.ckpt
#     torchrun --nnode=1 --nproc_per_node=${num_devices} --master_port ${random_port} src/train_dev2.py \
#         -mn $mn \
#         -pn amax_a100 \
#         -en $en\
#         -dn mqar \
#         --node_num 1 \
#         --device_num ${num_devices} \
#         --nworkers 0 \
#         --state train \
#         --lr_rate $lr \
#         --train_strategy "ddp" \
#         --monitor_metric "valid_lm_loss" \
#         --train_batch_size 120 \
#         --val_batch_size 64 \
#         --max_epochs 15 \
#         --module_type "longconv" \
#         --long_conv_kernel 128 \
#         --data_dir /nvme1/zecheng/data/MQAR/analysis/k2v2-standard;
# done

mn=mamba-tiny
en=mamba-tiny-position-standard-longconv-128-$lr
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
random_port=$(( (RANDOM % 10000) + 1024 ))
ckpt_dir=/nvme1/zecheng/ckpt/$en/$en/checkpoints
model_path=${ckpt_dir}/last.ckpt
python src/test_dev2.py \
    -mn $mn \
    -en ${en}-on_standard \
    -pn amax_a100 \
    -dn mqar \
    --state eval \
    --version $en \
    --val_batch_size 10 \
    --inference_mode \
    --nworkers 0 \
    --module_type "longconv" \
    --long_conv_kernel 128 \
    --ckpt_path /nvme1/zecheng/ckpt/mamba-tiny-position-standard-longconv-128-3e-3-post/version_1/checkpoints/last.ckpt \
    --data_dir /nvme1/zecheng/data/MQAR/analysis/position-standard/v0-standard;