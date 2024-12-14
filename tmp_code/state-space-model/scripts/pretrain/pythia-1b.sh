export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
random_port=$(( (RANDOM % 10000) + 1024 ))
torchrun --nnode=1 --nproc_per_node=${num_devices} --master_port ${random_port} src/train_dev2.py \
    -mn pythia-1b \
    -pn pjlab \
    -en pythia-1b-hf-2048-fromsk-15b \
    -dn slimpajama \
    --node_num 1 --max_seq_length 2048 --device_num ${num_devices} \
    --state train --save_top_k 5 --nworkers 0 --train_strategy "ddp" \
    --monitor_metric "valid_lm_loss" --accumulate_grad_batches 20 --max_epochs 10 \
    --lr_rate 1e-4 \
    --every_n_train_steps 400 \
    --train_batch_size 2 \
    --val_check_interval 400;

    # --ckpt_path /nvme1/zecheng/ckpt/pythia-160m-hf-2048-fromsk/version_1/checkpoints/last.ckpt;
    
    
    # \
    # --ckpt_path hf;


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
    