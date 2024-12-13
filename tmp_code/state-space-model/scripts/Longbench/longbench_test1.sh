export CUDA_VISIBLE_DEVICES=4

model_name=long_context_pythia
platform=amax_a100
task=longbench
model_path=/nvme1/zecheng/ckpt/longalpaca-long-context-pythia/version_1/checkpoints/last.ckpt
experiment_name=longbench

python src/test_dev2.py \
    -mn $model_name \
    -en $mark \
    -pn $platform \
    -dn $task \
    --state eval \
    --inference_mode
    # \
    # --ckpt_path $model_path;
    