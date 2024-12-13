export CUDA_VISIBLE_DEVICES=7
pretrained="/nvme/hf_models/mamba-130m-hf/"
ckpt="/nvme1/zecheng/ckpt/mamba-130m-hf-pretrain_dir/mamba-130m-hf-2048-from-sk-3b/version_1/checkpoints/last.ckpt"
lm_eval --model CustomMamba \
    --model_args "pretrained=$pretrained,ckpt=$ckpt"\
    --tasks squadv2 \
    --batch_size 10 \
    --log_samples \
    --device cuda \
    --limit 100 \
    --output_path /nvme1/zecheng/evaluation/lm_eval/$en \



# arc_easy,arc_challenge,lambada_openai,openbookqa,winogrande,piqa,hellaswag,wikitext 


# arc_easy,arc_challenge lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa 

# piqa  hellaswag,fda,swde

# squadv2 CustomMamba