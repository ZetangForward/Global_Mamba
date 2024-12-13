export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
random_port=$(( (RANDOM % 10000) + 1024 ))
en="mamba-1_4b-hf-flashlongconv-2048-2048"
ckpt_dir=/nvme1/zecheng/ckpt/$en/$en/checkpoints
model_path=${ckpt_dir}/last.ckpt
torchrun --nnode=1 --nproc_per_node=${num_devices} --master_port ${random_port} src/train_dev2.py \
    -mn mamba-1_4b-hf \
    -pn amax_a100 \
    -en $en \
    -dn slimpajama \
    --node_num 1 --max_seq_length 2048 --device_num ${num_devices} \
    --state train --save_top_k 5 --nworkers 0 --train_strategy "deepspeed" \
    --monitor_metric "valid_lm_loss" --accumulate_grad_batches 4 --max_epochs 2 \
    --lr_rate 8e-4 \
    --every_n_train_steps 400 \
    --train_batch_size 2 \
    --val_check_interval 400 \
    --long_conv_kernel 2048 \
    --module_type "flashlongconv" \
    --module_layer 7,15,23,31,39,47  \
    --ckpt_path hf;


# export CUDA_VISIBLE_DEVICES=0
# pretrained="/nvme/hf_models/mamba-130m-hf/"
# module_type=longconv
# long_conv_kernel=1024
# module_layers=11-23
# ckpt="/nvme1/zecheng/ckpt/mamba-130m-hf-longconv-2048-1024-1123-5e-5/version_1/checkpoints/last.ckpt"
# lm_eval --model CustomMamba \
#     --model_args "pretrained=$pretrained,ckpt=$ckpt,\
#                 module_layers=$module_layers,\
#                 module_type=$module_type,\
#                 long_conv_kernel=$long_conv_kernel"\
#     --tasks arc_easy,arc_challenge,lambada_openai,openbookqa,winogrande,piqa,hellaswag,wikitext \
#     --batch_size 10 \
#     --log_samples \
#     --device cuda \
#     --output_path /nvme1/zecheng/evaluation/lm_eval/$en \



    # --module_layer "23" ;

    # --module_layer "7,15,12,22,23" \
    # --ckpt_path hf ;

   

# torchrun --nnode=1 --nproc_per_node=${num_devices} src/train_dev2.py \
#     -mn mamba-130m-hf \
#     -pn amax_a100 \
#     -en mamba-130m-hf-slimpajama-lsgatedconv1d-128-inference \
#     -dn slimpajama \
#     --node_num 1 \
#     --device_num ${num_devices} \
#     --state eval \
#     --train_batch_size 9 \
#     --train_strategy "ddp" \
#     --monitor_metric "valid_lm_loss" \
#     --nworkers 0 \
#     --delta_ratio 0 \
#     --val_batch_size 1 \
#     --max_seq_len 2048 \
#     --max_epochs 1 \
#     --long_conv_kernel 128;

# python src/test_dev2.py \
#     -mn $model_name \
#     -en $mark \
#     -pn $platform \
#     -dn $task \
#     --state eval \
#     --inference_mode \
#     --ckpt_path $model_path;
    