export CUDA_VISIBLE_DEVICES=7
pretrained="/nvme/hf_models/mamba-130m-hf/"
ckpt="/nvme1/zecheng/ckpt/mamba-130m-hf-pretrain_dir/mamba-130m-hf-longconv-2048-512-71523-8e-4-fromsk-3b/version_1/checkpoints/last.ckpt"
module_type="longconv"
module_layers="7-15-23"
long_conv_kernel=512
lm_eval --model CustomMamba \
    --model_args "pretrained=$pretrained,ckpt=$ckpt,\
                module_layers=$module_layers,\
                module_type=$module_type,\
                long_conv_kernel=$long_conv_kernel"\
    --tasks squadv2\
    --batch_size 50 \
    --log_samples \
    --device cuda \
    --limit 100 \
    --output_path /nvme1/zecheng/evaluation/lm_eval/$en \


# fda,swde,squad_completion 
# arc_easy,arc_challenge,lambada_openai,openbookqa,winogrande,piqa,hellaswag,wikitext 


# arc_easy,arc_challenge lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa 

# piqa  hellaswag,fda,swde

# squadv2 CustomMamba