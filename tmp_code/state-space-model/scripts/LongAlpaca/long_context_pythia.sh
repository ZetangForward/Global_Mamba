export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

# torchrun --nnode=1 --nproc_per_node=3 
python src/train_dev2.py \
    -mn long_context_pythia \
    -pn amax_a100 \
    -en longalpaca-long-context-pythia \
    -dn longalpaca \
    --node_num 1 \
    --device_num 6 \
    --state train \
    --train_batch_size 1 \
    --val_batch_size 1 \
    --monitor_metric "train_lm_loss" \
    --save_top_k 5 \
    --every_n_train_steps 2000 \
    --accumulate_grad_batches 12 \
    --max_seq_len 32768 \
    --max_epochs 10 \
    --max_training_steps 20000 \
    --version version_1 \
    --hf_trainer;

