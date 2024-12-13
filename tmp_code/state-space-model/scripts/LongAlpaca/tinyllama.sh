export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# torchrun --nnode=1 --nproc_per_node=3 
python src/train_dev2.py \
    -mn tinyllama \
    -pn amax_a100 \
    -en longalpaca-long-tinyllama \
    -dn longalpaca \
    --node_num 1 \
    --device_num 8 \
    --nworkers 4 \
    --state train \
    --train_batch_size 2 \
    --train_strategy "fsdp" \
    --monitor_metric "train_lm_loss" \
    --save_top_k 5 \
    --every_n_train_steps 50 \
    --accumulate_grad_batches 8 \
    --max_seq_len 16384 \
    --max_epochs 15 \
    --max_training_steps 20000 \
    --version version_1 \
    --hf_trainer \

