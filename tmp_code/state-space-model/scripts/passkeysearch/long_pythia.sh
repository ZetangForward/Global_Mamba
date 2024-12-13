export CUDA_VISIBLE_DEVICES=0,1,2,3

python src/test_dev2.py \
-mn long_gpt_neo \
-pn langchao \
-en test \
--max_seq_len 128000 \
--data_name "passkey_search" \
--ckpt_path "/public/home/ljt/tzc/ckpt/longalpaca-long-context-pythia/version_1/checkpoints/last.ckpt" \