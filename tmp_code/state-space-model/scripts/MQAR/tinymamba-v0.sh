export CUDA_VISIBLE_DEVICES=6,7
mn=mamba-130m-hf
en=mamba-130m-v0-standard-for_tiny
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
    --nworkers 4 \
    --state train \
    --train_strategy "ddp" \
    --monitor_metric "valid_lm_loss" \
    --train_batch_size 96 \
    --val_batch_size 32 \
    --max_epochs 30 \
    --lr_rate 1e-3 \
    --version $en \
    --data_dir "MQAR/for_tiny/mqar-v0-standard";
#     --data_dir "MQAR/for_tiny/mqar-v0-standard";
# /nvme1/zecheng/data/MQAR/analysis/position-standard/v0-standard
# wait 

# python ${ckpt_dir}/last.ckpt/zero_to_fp32.py ${ckpt_dir}/last.ckpt $model_path

# wait

# python src/test_dev2.py \
#     -mn $mn \
#     -en ${en}_4k_test \
#     -pn amax_a100 \
#     -dn mqar \
#     --state eval \
#     --version $en \
#     --val_batch_size 1 \
#     --inference_mode \
#     --max_seq_len 16384 \
#     --nworkers 0 \
#     --data_dir MQAR/mqar-v0-k1v1-512-for_test \
#     --delta_ratio 0 \
#     --long_conv_kernel 128 \
#     --ckpt_path /nvme1/zecheng/ckpt/tinymamba/tiny_mamba-k4-l2-tsr16-s16-v0-512/tiny_mamba-k4-l2-tsr16-s16-v0-512/checkpoints/last.ckpt ;
    

