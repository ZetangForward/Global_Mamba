export CUDA_VISIBLE_DEVICES=1

python src/test_dev2.py \
-mn mamba-370m-k8 \
-pn amax_a100 \
-en test \
--data_name "passkey_search" \
--ckpt_path "/nvme/zecheng/ckpt/h_800/ckpt/slimpajama/mamba_370m_big_kernel-k8/checkpoints/last.ckpt/model.bin" \