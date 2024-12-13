export CUDA_VISIBLE_DEVICES=0,1

platform=amax_a100
task=longbench
model_path=/nvme1/zecheng/ckpt/longalpaca-long-context-pythia/version_1/checkpoints/last.ckpt
experiment_name=longbench

python src/test_dev2.py \
    -mn long_gpt_neo \
    -en longbench \
    -pn $platform \
    -dn $task \
    --nworkers 0 \
    --device_num 8 \
    --state eval \
    --val_batch_size 8 \
    --inference_mode \
    --ckpt_path $model_path;
    # --train_strategy 'fsdp' \
    # --model_module models.long_pythia \
    # --block_name GPTNeoBlock \
    
    