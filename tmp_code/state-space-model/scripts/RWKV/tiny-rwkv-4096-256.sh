export CUDA_VISIBLE_DEVICES=0,1,2,3
bash prepare.sh
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

python src/train_dev2.py \
    -mn tiny-rwkv-169m \
    -pn amax_a100 \
    -en tiny-rwkv-4096-256 \
    -dn MQAR \
    --processed_data_path MQAR/train_based_in.pkl \
    --node_num 1 \
    --state train \
    --train_batch_size 64 \
    --val_batch_size 1 \
    --max_epochs 50 \
    --num_kv_pairs 0 \
    --max_seq_len 8 \
    --monitor_metric "valid_lm_loss" \
    --version tzc \
    --lr_rate 5e-5 \
    --n_layer 4 --n_embd 256 --pre_ffn 0 --head_qk 0 \
    --vocab_size 20480 \
    --weight_decay 0.01 \
    --accelerator gpu --precision bf16 \
    --dropout 0.05 \
    --my_testing x052;




