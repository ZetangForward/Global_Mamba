export CUDA_VISIBLE_DEVICES=7
pretrained="/nvme/hf_models/mamba-130m-hf/"
ckpt="/nvme1/zecheng/ckpt/mamba-130m-hf-pretrain_dir/mamba-130m-hf-2048-from-sk-3b/version_1/checkpoints/last.ckpt"
lm_eval --model CustomMamba \
    --model_args "pretrained=$pretrained,ckpt=$ckpt"\
    --tasks squadv2 \
    --batch_size 30 \
    --log_samples \
    --device cuda \
    --output_path /nvme1/zecheng/evaluation/lm_eval/$en \

wait

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
    --batch_size 30 \
    --log_samples \
    --device cuda \
    --output_path /nvme1/zecheng/evaluation/lm_eval/$en \

pretrained="/nvme/hf_models/pythia-160m"
lm_eval --model hf \
    --model_args "pretrained=$pretrained"\
    --tasks squadv2\
    --batch_size 10 \
    --log_samples \
    --device cuda \
    --output_path /nvme1/zecheng/evaluation/lm_eval/$en \

wait

pretrained="/nvme/hf_models/pythia-410m"
ckpt="/nvme1/zecheng/ckpt/pythia-410m-hf-2048-fromsk-15b/best_model.bin"
lm_eval --model CustomMamba \
    --model_args "pretrained=$pretrained,ckpt=$ckpt"\
    --tasks squadv2 \
    --batch_size 30 \
    --show_config \
    --log_samples \
    --cache_requests true \
    --device cuda \
    --output_path /nvme1/zecheng/evaluation/lm_eval/ \

wait

pretrained="/nvme/hf_models/mamba-370m-hf/"
ckpt="/nvme1/zecheng/ckpt/mamba-370m-hf-2048-from-sk-15b_final_1/version_1/checkpoints/last.ckpt"
lm_eval --model CustomMamba \
    --model_args "pretrained=$pretrained,ckpt=$ckpt" \
    --tasks squadv2 \
    --batch_size 30 \
    --log_samples \
    --device cuda \
    --output_path /nvme1/zecheng/evaluation/lm_eval/$en \

wait

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
    --batch_size 30 \
    --log_samples \
    --device cuda \
    --output_path /nvme1/zecheng/evaluation/lm_eval/$en \

wait

pretrained="/nvme/hf_models/mamba-370m-hf/"
ckpt="/nvme1/zecheng/ckpt/mamba-370m-hf-2048-from-sk-15b_final_1/version_1/checkpoints/last.ckpt"
lm_eval --model CustomMamba \
    --model_args "pretrained=$pretrained,ckpt=$ckpt" \
    --tasks scrolls_qasper,scrolls_qmsum \
    --batch_size 15 \
    --log_samples \
    --device cuda \
    --output_path /nvme1/zecheng/evaluation/lm_eval/$en \

wait

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
    --tasks scrolls_qasper,scrolls_qmsum \
    --batch_size 15 \
    --log_samples \
    --device cuda \
    --output_path /nvme1/zecheng/evaluation/lm_eval/$en \

wait

pretrained="/nvme/hf_models/mamba-130m-hf/"
ckpt="/nvme1/zecheng/ckpt/mamba-130m-hf-pretrain_dir/mamba-130m-hf-2048-from-sk-3b/version_1/checkpoints/last.ckpt"
lm_eval --model CustomMamba \
    --model_args "pretrained=$pretrained,ckpt=$ckpt"\
    --tasks scrolls_qasper,scrolls_qmsum \
    --batch_size 30 \
    --log_samples \
    --device cuda \
    --output_path /nvme1/zecheng/evaluation/lm_eval/$en \

wait

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
    --tasks scrolls_qasper,scrolls_qmsum\
    --batch_size 30 \
    --log_samples \
    --device cuda \
    --output_path /nvme1/zecheng/evaluation/lm_eval/$en \