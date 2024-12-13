export CUDA_VISIBLE_DEVICES=0
mn=mamba-130m-hf
en=mamba-130m-v6-k2v2-512-seq_loss-lsgatedconv
# mamba-130m-v6-k4v8-512-seq_loss-lsgatedconv1d-128-nodecay-noconvgated
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
#     --val_batch_size 12 \
#     --max_epochs 30 \
#     --version $en \
#     --long_conv_kernel 32 \
#     --module_type longconv \
#     --data_dir MQAR/mqar-v6-k4v8-512-seq_loss ;

python src/test_dev2.py \
    -mn $mn \
    -en ${en} \
    -pn amax_a100 \
    -dn mqar \
    --state eval \
    --version $en \
    --val_batch_size 1 \
    --inference_mode \
    --max_seq_len 16384 \
    --nworkers 0 \
    --ckpt_path /nvme1/zecheng/ckpt/mamba-130m-v6-k2v2-512-seq_loss-lsgatedconv1d-76/mamba-130m-v6-k2v2-512-seq_loss-lsgatedconv1d-76/checkpoints/epoch=29-step=75000-valid_lm_loss=0.00.ckpt \
    --long_conv_kernel 128 \
    --data_dir /nvme1/zecheng/data/MQAR/mqar-v6-k2v2-512-seq_loss \


    # --test_path /nvme1/zecheng/data/MQAR/analysis/mqar-v6-k2v2-standard-robustness/test.jsonl \

