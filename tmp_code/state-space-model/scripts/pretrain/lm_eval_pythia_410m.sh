export CUDA_VISIBLE_DEVICES=7
pretrained="/nvme/hf_models/pythia-410m"
ckpt="/nvme1/zecheng/ckpt/pythia-410m-hf-2048-fromsk-15b/best_model.bin"
lm_eval --model CustomMamba \
    --model_args "pretrained=$pretrained,ckpt=$ckpt"\
    --tasks squadv2 \
    --batch_size 10 \
    --show_config \
    --log_samples \
    --cache_requests true \
    --limit 100 \
    --device cuda \
    --output_path /nvme1/zecheng/evaluation/lm_eval/ \


# export CUDA_VISIBLE_DEVICES=0
# pretrained="/nvme/hf_models/pythia-410m"
# lm_eval --model hf \
#     --model_args "pretrained=$pretrained"\
#     --tasks squad_completion \
#     --batch_size 20 \
#     --show_config \
#     --log_samples \
#     --cache_requests true \
#     --device cuda \
#     --output_path /nvme1/zecheng/evaluation/lm_eval/ \



# arc_easy,arc_challenge lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa 

# piqa  hellaswag
#  fda,swde,squadv2
# squadv2 CustomMamba