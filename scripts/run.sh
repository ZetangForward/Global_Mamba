### For original Mamba
export CUDA_VISIBLE_DEVICES=
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
random_port=$(( (RANDOM % 10000) + 1024 ))
mn=mamba-130m-hf
en=
ckpt_dir=
model_path=${ckpt_dir}/last.ckpt
torchrun --nnode=1 --nproc_per_node=${num_devices} --master_port ${random_port} src/train_dev2.py \
    -mn $mn \
    -pn your_filepath \
    -en $en\
    -dn mqar \
    --node_num 1 \
    --device_num ${num_devices} \
    --nworkers 0 \
    --state train \
    --train_strategy "ddp" \
    --monitor_metric "valid_lm_loss" \
    --train_batch_size 1 \
    --val_batch_size 1 \
    --max_epochs 30 \
    --version $en \
    --data_dir 


python src/test_dev2.py \
    -mn $mn \
    -en ${en} \
    -pn amax_a100 \
    -dn mqar \
    --state eval \
    --version $en \
    --val_batch_size 1 \
    --inference_mode \
    --ckpt_path ${model_path} \
    --data_dir your_data_dir;



## For Gobal Selection 
## add
# --long_conv_kernel 128 \
# --module_type longconv \