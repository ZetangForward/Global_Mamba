export CUDA_VISIBLE_DEVICES=$1

num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
platform=langchao
id=$2

model_path=/public/home/ljt/tzc/ckpt/Copy-mamba-370m-s${id}/1/checkpoints/last.ckpt

for ((input_len=4; input_len<=512; input_len+=4));
do
    mark=Copy-mamba-370m-s${id}-len${input_len}
    python src/test_dev2.py \
        -mn mamba-370m-s${id} \
        --ckpt_path $model_path \
        -en $mark \
        -pn $platform \
        -dn Copy \
        --state eval \
        --version 0 \
        --inference_mode \
        --max_seq_len $input_len \
        --processed_data_path Copy/V4096_N100_test/test_V4096_L${input_len}.pkl
done

