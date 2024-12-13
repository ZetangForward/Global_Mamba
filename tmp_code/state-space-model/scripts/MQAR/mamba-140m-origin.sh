export CUDA_VISIBLE_DEVICES=7
mn=mamba-790m-hf
en=mamba-790m-hf
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
random_port=$(( (RANDOM % 10000) + 1024 ))
torchrun --nnode=1 --nproc_per_node=${num_devices} --master_port ${random_port} src/train_dev2.py \
    -mn $mn \
    -pn h20 \
    -en $en\
    -dn mqar \
    --node_num 1 \
    --device_num ${num_devices} \
    --nworkers 0 \
    --state train \
    --lr_rate 3e-3 \
    --train_strategy "ddp" \
    --monitor_metric "valid_lm_loss" \
    --train_batch_size 20 \
    --val_batch_size 12 \
    --max_epochs 1 \
    --version $en \
    --module_type flashlongconv \
    --long_conv_kernel 2048 \
    --module_layer 7,15,23,31,39,47  \
    --data_dir /nvme/ywj/data/robustness_v0_position ;


# python src/test_dev2.py \
#     -mn $mn \
#     -en ${en}_longconv_k2v2_standard_on_k2v2_standard \
#     -pn amax_a100 \
#     -dn mqar \
#     --state eval \
#     --version $en \
#     --val_batch_size 1 \
#     --inference_mode \
#     --max_seq_len 16384 \
#     --nworkers 0 \
#     --module_type longconv \
#     --long_conv_kernel 128 \
#     --ckpt_path /nvme1/zecheng/ckpt/mamba-130m-hf-longconv-mqar/mamba-130m-v6-k2v2-standard-longconv-128/mamba-130m-v6-k2v2-standard-longconv-128/checkpoints/last.ckpt \
#     --data_dir /nvme1/zecheng/data/MQAR/analysis/k2v2-standard;
    
# python src/test_dev2.py \
#     -mn $mn \
#     -en ${en}-etetsetstetset \
#     -pn amax_a100 \
#     -dn mqar \
#     --state eval \
#     --version $en \
#     --val_batch_size 1 \
#     --inference_mode \
#     --max_seq_len 16384 \
#     --nworkers 0 \
#     --ckpt_path /nvme1/zecheng/ckpt/analysis-pre/mamba-130m-hf-v6-k2v2-standard/mamba-130m-hf-v6-k2v2-standard/checkpoints/last.ckpt \
#     --test_path /nvme1/zecheng/data/MQAR/position-last.jsonl;

#  python src/test_dev2.py \
#     -mn $mn \
#     -en ${en}-etetsetstetset \
#     -pn amax_a100 \
#     -dn mqar \
#     --state eval \
#     --version $en \
#     --val_batch_size 1 \
#     --inference_mode \
#     --max_seq_len 16384 \
#     --nworkers 0 \
#     --ckpt_path /nvme1/zecheng/ckpt/analysis-pre/mamba-130m-hf-v6-k2v2-standard/mamba-130m-hf-v6-k2v2-standard/checkpoints/last.ckpt \
#     --test_path /nvme1/zecheng/data/MQAR/position-shuffle.jsonl;

   



