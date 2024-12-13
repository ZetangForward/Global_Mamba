export CUDA_VISIBLE_DEVICES=4
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
id=1

model_name=mamba-370m-km
platform=hitsz
task=longbench_ywj
model_path=/UNICOMFS/hitsz_khchen_4/zecheng/ckpt/longalpaca/version_1/checkpoints/last-v1.ckpt
mark=longalpaca

mn=mamba-1.4b-hf
en=mamba-130m-v6-k2v2-512-seq_loss-lsgatedconv1d-origin-64-728
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
random_port=$(( (RANDOM % 10000) + 1024 ))
ckpt_dir=/nvme1/zecheng/ckpt/$en/$en/checkpoints
model_path=${ckpt_dir}/last.ckpt

python src/test_dev2.py \
    -mn $model_name \
    -en $mark \
    -pn $platform \
    -dn $task \
    --state eval \
    --inference_mode \
    --ckpt_path $model_path;
    

