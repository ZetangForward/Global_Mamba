export CUDA_VISIBLE_DEVICES=
mn=mamba-130m-hf
en=your_expname
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
random_port=$(( (RANDOM % 10000) + 1024 ))
ckpt_dir=your_ckptpath
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
    --lr_rate 1e-3 \
    --train_strategy "ddp" \
    --monitor_metric "valid_lm_loss" \
    --train_batch_size 1 \
    --val_batch_size 1 \
    --max_epochs 30 \
    --version $en \
    --long_conv_kernel 128 \
    --module_type longconv \
    --data_dir your_data_dir;


python src/test_dev2.py \
    -mn $mn \
    -en ${en}-test \
    -pn amax_a100 \
    -dn mqar \
    --state eval \
    --version $en \
    --val_batch_size 1 \
    --inference_mode \
    --nworkers 0 \
    --ckpt_path ${model_path} \
    --long_conv_kernel 128 \
    --module_type "longconv" \
    --data_dir your_data_dir;
