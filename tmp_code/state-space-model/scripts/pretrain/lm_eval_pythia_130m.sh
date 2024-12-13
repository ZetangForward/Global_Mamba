export CUDA_VISIBLE_DEVICES=7
# pretrained="/nvme1/zecheng/ckpt/pythia_dir/pythia-130m-hf-2048-fromsk-3b-8e-4/hf_version"
pretrained="/nvme/hf_models/pythia-160m"
# ,ckpt=$ckpt
# ckpt="/nvme1/zecheng/ckpt/pythia_dir/pythia-130m-hf-2048-fromsk-3b-8e-4/version_1/checkpoints/last.ckpt"
# # ,ckpt=$ckpt"
lm_eval --model hf \
    --model_args "pretrained=$pretrained"\
    --tasks squadv2\
    --batch_size 10 \
    --log_samples \
    --limit 100 \
    --device cuda \
    --output_path /nvme1/zecheng/evaluation/lm_eval/$en \


# scrolls_narrativeqa
#  scrolls_govreport,scrolls_qasper,

# export CUDA_VISIBLE_DEVICES=1
# pretrained="/nvme/hf_models/pythia-160m"
# ckpt="/nvme1/zecheng/ckpt/pythia_dir/pythia-130m-hf-2048-fromsk-3b-8e-4/version_1/checkpoints/last.ckpt"
# # ,ckpt=$ckpt"
# lm_eval --model CustomMamba \
#     --model_args "pretrained=$pretrained,ckpt=$ckpt"\
#     --tasks  squad_completion \
#     --batch_size 1 \
#     --log_samples \
#     --device cuda \
#     --limit 10 \
#     --output_path /nvme1/zecheng/evaluation/lm_eval/$en \





# arc_easy,arc_challenge,lambada_openai,openbookqa,winogrande,piqa,hellaswag,wikitext

# arc_easy,arc_challenge lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa 

# piqa  hellaswag,fda,swde

# squadv2 CustomMamba