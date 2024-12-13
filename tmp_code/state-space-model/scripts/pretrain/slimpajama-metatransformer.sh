export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

torchrun --nnode=1 --nproc_per_node=6 src/train_dev2.py \
    -mn metatransformer \
    -pn amax_a100 \
    -en metatransformer \
    -dn slimpajama \
    --node_num 1 \
    --state train \
    --train_batch_size 6 \
    --val_batch_size 6 \
    --monitor_metric "train_lm_loss" \
    --save_top_k 5 \
    --every_n_train_steps 2000 \
    --accumulate_grad_batches 4 \
    --max_seq_len 4000 \
    --max_training_steps 20000 \
    --use_deepspeed \
    --version version_1 \
    --hf_trainer;

