export CUDA_VISIBLE_DEVICES=$1
num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
platform=langchao
id=$2

mark=tuned
model_path=/public/home/ljt/tzc/ckpt/Copy-mamba-370m-${id}-lr1e-3/version_1/checkpoints/last.ckpt
python custom_mamba/copy_analysis.py \
    -mn mamba-370m-${id} \
    -en $mark \
    -pn $platform \
    -dn Copy \
    --ckpt_path ${model_path};




   