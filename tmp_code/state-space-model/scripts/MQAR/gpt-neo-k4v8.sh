export CUDA_VISIBLE_DEVICES=0,1,2,3
mn=pythia-160m
en=pythia-130m-k1v2-standard
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
    --train_batch_size 50 \
    --val_batch_size 12 \
    --max_epochs 30 \
    --version $en \
    --hf_trainer \
    --data_dir MQAR/analysis/k1v2-standard ;

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
    --ckpt_path ${model_path} \
    --data_dir MQAR/analysis/k1v2-standard;