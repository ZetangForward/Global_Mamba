export CUDA_VISIBLE_DEVICES=7
pretrained="/nvme/hf_models/mamba-370m-hf/"
ckpt="/nvme1/zecheng/ckpt/mamba-370m-hf-2048-from-sk-15b_final_1/version_1/checkpoints/last.ckpt"
lm_eval --model CustomMamba \
    --model_args "pretrained=$pretrained,ckpt=$ckpt" \
    --tasks squadv2 \
    --batch_size 10 \
    --log_samples \
    --limit 100 \
    --device cuda \
    --output_path /nvme1/zecheng/evaluation/lm_eval/$en \



# arc_easy,arc_challenge,lambada_openai,openbookqa,winogrande,piqa,hellaswag,wikitext
# arc_easy,arc_challenge,lambada_openai,openbookqa,winogrande,piqa,hellaswag,wikitext

# arc_easy,arc_challenge lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa 

# piqa  hellaswag,fda,swde

# squadv2 CustomMamba