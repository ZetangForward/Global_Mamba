export CUDA_VISIBLE_DEVICES=7
pretrained="/nvme/hf_models/mamba-370m-hf/"
module_type=longconv
long_conv_kernel=512
module_layers=7-15-23-31-39-47 
ckpt="/nvme1/zecheng/ckpt/mamba-370m-hf-pretrain/mamba-370m-hf-longconv-2048-512--from-sk-15b/version_1/checkpoints/last.ckpt"
lm_eval --model CustomMamba \
    --model_args "pretrained=$pretrained,ckpt=$ckpt,\
                module_layers=$module_layers,\
                module_type=$module_type,\
                long_conv_kernel=$long_conv_kernel"\
    --tasks squadv2 \
    --batch_size 50 \
    --log_samples \
    --device cuda \
    --limit 100 \
    --output_path /nvme1/zecheng/evaluation/lm_eval/$en \

# ,fda,swde


# arc_easy,arc_challenge,lambada_openai,openbookqa,winogrande,piqa,hellaswag,wikitext


# arc_easy,arc_challenge lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa 

# piqa  hellaswag,fda,swde

# squadv2 CustomMamba