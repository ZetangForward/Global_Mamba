export CUDA_VISIBLE_DEVICES=0
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
random_port=$(( (RANDOM % 10000) + 1024 ))
en=mamba-130m-longconv_eff
torchrun --nnode=1 --nproc_per_node=${num_devices} --master_port ${random_port} src/eff_benchmark.py \
    -mn mamba-130m-hf \
    -pn amax_a100 \
    -en $en \
    -dn slimpajama \
    --node_num 1 --max_seq_length 2048 --device_num ${num_devices} \
    --state train \
    --train_batch_size 8 \
    --long_conv_kernel 512 \
    --module_type "longconv" \
    --module_layer 7,15,23  \

