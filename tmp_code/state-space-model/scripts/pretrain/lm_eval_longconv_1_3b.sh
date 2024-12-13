export CUDA_VISIBLE_DEVICES=7
pretrained="/nvme/hf_models/mamba-1.4b-hf/"
module_type=flashlongconv
long_conv_kernel=2048
module_layers=7-15-23-31-39-47
ckpt="/nvme1/zecheng/ckpt/mamba-1_4b-hf-flashlongconv-2048-2048/version_1/checkpoints/last.ckpt/model.bin"
# lm_eval --model CustomMamba \
#     --model_args "pretrained=$pretrained,ckpt=$ckpt,\
#                 module_layers=$module_layers,\
#                 module_type=$module_type,\
#                 long_conv_kernel=$long_conv_kernel"\
#     --tasks arc_easy,arc_challenge,lambada_openai,openbookqa,winogrande,piqa,hellaswag,wikitext,fda,swde,squad_completion \
#     --batch_size 10 \
#     --log_samples \
#     --device cuda \
#     --output_path /nvme1/zecheng/evaluation/lm_eval/$en \

# wait

export CUDA_VISIBLE_DEVICES=7
pretrained="/nvme/hf_models/mamba-1.4b-hf/"
ckpt="/nvme/hf_models/mamba-1.4b-hf/"
lm_eval --model CustomMamba \
    --model_args "pretrained=$pretrained,ckpt=$ckpt"\
    --tasks arc_easy,arc_challenge,lambada_openai,openbookqa,winogrande,piqa,hellaswag,wikitext,fda,swde,squad_completion \
    --batch_size 10 \
    --log_samples \
    --device cuda \
    --output_path /nvme1/zecheng/evaluation/lm_eval/$en \




# arc_easy,arc_challenge,lambada_openai,openbookqa,winogrande,piqa,hellaswag,wikitext


# arc_easy,arc_challenge lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa 

# piqa  hellaswag,fda,swde

# squadv2 CustomMamba