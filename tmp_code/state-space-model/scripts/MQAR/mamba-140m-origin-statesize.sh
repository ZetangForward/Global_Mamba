export CUDA_VISIBLE_DEVICES=4,5,6,7
state_size=128
mn=mamba-130m-hf
en=mamba-130m-hf-v0-position-standard-statesize${state_size}
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
    --lr_rate 8e-4 \
    --train_strategy "ddp" \
    --monitor_metric "valid_lm_loss" \
    --train_batch_size 50 \
    --val_batch_size 12 \
    --max_epochs 30 \
    --version $en \
    --state_size ${state_size} \
    --n_layers 24 \
    --data_dir MQAR/analysis/position-standard/v0-standard;

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
    --ckpt_path ${model_path} \
    --state_size ${state_size} \
    --n_layers 24 \
    --data_dir MQAR/analysis/position-standard/v0-standard;


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
#     --state_size ${state_size} \
#     --n_layers 24 \
#     --test_path /nvme1/zecheng/data/MQAR/analysis/mqar-v6-k2v2-standard-robustness/test.jsonl;


# python src/test_dev2.py \
#     -mn $mn \
#     -en ${en}_on_standard \
#     -pn amax_a100 \
#     -dn mqar \
#     --state eval \
#     --version $en \
#     --val_batch_size 1 \
#     --inference_mode \
#     --max_seq_len 16384 \
#     --nworkers 0 \
#     --ckpt_path ${model_path} \
#     --state_size 32 \
#     --n_layers 24 \
#     --data_dir MQAR/analysis/position-standard/v0-standard;

# python src/test_dev2.py \
#     -mn $mn \
#     -en ${en}_on_last \
#     -pn amax_a100 \
#     -dn mqar \
#     --state eval \
#     --version $en \
#     --val_batch_size 1 \
#     --inference_mode \
#     --max_seq_len 16384 \
#     --nworkers 0 \
#     --ckpt_path ${model_path} \
#     --state_size 64 \
#     --n_layers 24 \
#     --data_dir MQAR/analysis/position-last/v0-last;

# python src/test_dev2.py \
#     -mn $mn \
#     -en ${en}_on_shuffle \
#     -pn amax_a100 \
#     -dn mqar \
#     --state eval \
#     --version $en \
#     --val_batch_size 1 \
#     --inference_mode \
#     --max_seq_len 16384 \
#     --nworkers 0 \
#     --ckpt_path ${model_path} \
#     --state_size 64 \
#     --n_layers 24 \
#     --data_dir MQAR/analysis/position-shuffle/v0-shuffle;

