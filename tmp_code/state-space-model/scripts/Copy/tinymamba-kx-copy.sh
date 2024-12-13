export CUDA_VISIBLE_DEVICES=$1
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
id=$2
random_port=$(( (RANDOM % 10000) + 1024 ))

lr_list="1e-3"
for lr in $lr_list
do
    torchrun --nnode=1 --nproc_per_node=$num_devices --master_port ${random_port} src/train_dev2.py \
        -mn tiny_mamba-k${id} \
        -pn langchao \
        -en Copy-tinymamba-k${id}-lr${lr} \
        -dn Copy \
        --node_num 1 \
        --device_num ${num_devices} \
        --state train \
        --train_batch_size 256 \
        --max_epochs 50 \
        --max_seq_len 512 \
        --monitor_metric "valid_lm_loss" \
        --version 1 \
        --lr_rate $lr;
done


