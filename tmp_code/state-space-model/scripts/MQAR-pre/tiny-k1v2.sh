export CUDA_VISIBLE_DEVICES=4,5,6,7
mn=mamba-130m-hf
en=mamba-130m-hf-v6-k1v2-standard
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
    --train_batch_size 44 \
    --val_batch_size 32 \
    --max_epochs 30 \
    --version $en \
    --delta_ratio 0 \
    --data_dir MQAR/analysis/k1v2-standard;

wait 

python src/test_dev2.py \
    -mn $mn \
    -en ${en}_test_on_k1v1 \
    -pn amax_a100 \
    -dn mqar \
    --state eval \
    --version $en \
    --val_batch_size 1 \
    --inference_mode \
    --max_seq_len 16384 \
    --nworkers 0 \
    --delta_ratio 0 \
    --ckpt_path $model_path\
    --data_dir MQAR/analysis/k1v1-standard;
    
wait

python src/test_dev2.py \
    -mn $mn \
    -en ${en}_test_on_k2v2 \
    -pn amax_a100 \
    -dn mqar \
    --state eval \
    --version $en \
    --val_batch_size 1 \
    --inference_mode \
    --max_seq_len 16384 \
    --nworkers 0 \
    --delta_ratio 0 \
    --ckpt_path $model_path\
    --data_dir MQAR/analysis/k2v2-standard;

wait 

python src/test_dev2.py \
    -mn $mn \
    -en ${en}_test_on_k1v2 \
    -pn amax_a100 \
    -dn mqar \
    --state eval \
    --version $en \
    --val_batch_size 1 \
    --inference_mode \
    --max_seq_len 16384 \
    --nworkers 0 \
    --delta_ratio 0 \
    --ckpt_path $model_path\
    --data_dir MQAR/analysis/k1v2-standard;

wait 

python src/test_dev2.py \
    -mn $mn \
    -en ${en}_test_on_k2v4 \
    -pn amax_a100 \
    -dn mqar \
    --state eval \
    --version $en \
    --val_batch_size 1 \
    --inference_mode \
    --max_seq_len 16384 \
    --nworkers 0 \
    --delta_ratio 0 \
    --ckpt_path $model_path\
    --data_dir MQAR/analysis/k2v4-standard;