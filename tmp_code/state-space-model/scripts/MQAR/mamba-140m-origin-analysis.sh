export CUDA_VISIBLE_DEVICES=7
mn=mamba-130m-hf
en=mamba-130m-v6-k2v2-longconv-robustness
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
random_port=$(( (RANDOM % 10000) + 1024 ))
ckpt_dir=/nvme1/zecheng/ckpt/$en/$en/checkpoints
model_path=${ckpt_dir}/last.ckpt


python src/test_dev2.py \
    -mn $mn \
    -en ${en}-test \
    -pn amax_a100 \
    -dn mqar \
    --state eval \
    --version $en \
    --val_batch_size 1 \
    --inference_mode \
    --max_seq_len 16384 \
    --nworkers 0 \
    --test_path /nvme1/zecheng/data/MQAR/analysis/mqar-v6-k2v2-standard-robustness/test.jsonl \
    --ckpt_path /nvme1/zecheng/ckpt/analysis-pre/mamba-130m-hf-v6-k2v2-standard/mamba-130m-hf-v6-k2v2-standard/checkpoints/last.ckpt;

    # --module_type "origin" \
    # --record_debug_params \