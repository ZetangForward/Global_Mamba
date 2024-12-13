export CUDA_VISIBLE_DEVICES=$1
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
id=$2
random_port=$(( (RANDOM % 10000) + 1024 ))

lr_list="1e-3"
for lr in $lr_list
do
    torchrun --nnode=1 --nproc_per_node=$num_devices --master_port ${random_port} src/train_dev2.py \
        -mn mamba-370m-s${id} \
        -pn langchao \
        -en Copy-mamba-370m-s${id} \
        -dn Copy \
        --node_num 1 \
        --device_num ${num_devices} \
        --state train \
        --train_batch_size 16 \
        --max_epochs 50 \
        --monitor_metric "valid_lm_loss" \
        --version 1 \
        --lr_rate $lr \
        --processed_data_path "Copy/V4096_train.pkl";
done


