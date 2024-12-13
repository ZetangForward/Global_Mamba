export CUDA_VISIBLE_DEVICES=7
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
random_port=$(( (RANDOM % 10000) + 1024 ))
en="mamba-130m-hf-test"
ckpt_dir=/nvme1/zecheng/ckpt/version_1/checkpoints
model_path=${ckpt_dir}/last.ckpt
torchrun --nnode=1 --nproc_per_node=${num_devices} --master_port ${random_port} src/train_dev2.py \
    -mn mamba-130m-hf \
    -pn amax_a100 \
    -en $en \
    -dn slimpajama \
    --node_num 1 --max_seq_length 2048 --device_num ${num_devices} \
    --state train --save_top_k 5 --nworkers 0 --train_strategy "ddp" \
    --monitor_metric "valid_lm_loss" --accumulate_grad_batches 4 --max_epochs 1 \
    --lr_rate 5e-4 \
    --every_n_train_steps 400 \
    --train_batch_size 1 \
    --val_check_interval 400 \
    --ckpt_path /nvme1/zecheng/ckpt/mamba-130m-hf-2048-from-sk/version_1/checkpoints/last.ckpt;



# export CUDA_VISIBLE_DEVICES=0
# pretrained="/nvme/hf_models/mamba-130m-hf/"
# module_type=longconv
# long_conv_kernel=1024
# module_layers=11-23
# ckpt=${model_path}

# lm_eval --model CustomMamba \
#     --model_args "pretrained=$pretrained,ckpt=$ckpt,\

#     --tasks arc_easy,arc_challenge,lambada_openai,openbookqa,winogrande,piqa,hellaswag,wikitext \
#     --batch_size 10 \
#     --log_samples \
#     --device cuda \
#     --output_path /nvme1/zecheng/evaluation/lm_eval/$en \

