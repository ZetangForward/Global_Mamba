export CUDA_VISIBLE_DEVICES=1
mn=mamba-130m-hf
en=mamba-130m-v6-k4v8-longconv-128-60epoch
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
#     --train_batch_size 38 \
#     --val_batch_size 12 \
#     --max_epochs 60 \
#     --version $en \
#     --long_conv_kernel 128 \
#     --module_type longconv \
#     --data_dir /nvme1/zecheng/data/MQAR/mqar-v6-k4v8-512-seq_loss;

    # --accumulate_grad_batches 2 \

python src/test_dev2.py \
    -mn $mn \
    -en ${en}-test \
    -pn amax_a100 \
    -dn mqar \
    --state eval \
    --version $en \
    --val_batch_size 1 \
    --inference_mode \
    --max_seq_len 16384 \
    --nworkers 0 \
    --ckpt_path /nvme1/zecheng/ckpt/mamba-130m-hf-pretrain_dir/mamba-130m-hf-longconv-2048-512-71523-8e-4-fromsk-3b/version_1/checkpoints/last.ckpt \
    --module_layer 7,15,23 \
    --module_type "longconv" \
    --long_conv_kernel 512 \
    --data_dir /nvme1/zecheng/data/MQAR/mqar-v6-k4v8-512-seq_loss;



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