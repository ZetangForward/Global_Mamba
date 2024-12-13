export CUDA_VISIBLE_DEVICES=0
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

python src/test_dev2.py \
    -mn mamba-130m-hf \
    -en mamba-130m-hf-slimpajama \
    -pn amax_a100 \
    -dn pg19 \
    --ckpt_path /nvme1/zecheng/ckpt/mamba-130m-hf-from-sk/version_1/checkpoints/epoch=0-step=45600-train_lm_loss=3.00.ckpt \


# export CUDA_VISIBLE_DEVICES=5
# num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
# python src/test_dev2.py \
#     -mn mamba-130m-hf \
#     -en mamba-130m-hf-slimpajama-origin-test \
#     -pn amax_a100 \
#     -dn pg19 \
#     --max_seq_len 1000000 \
#     --ckpt_path hf;

