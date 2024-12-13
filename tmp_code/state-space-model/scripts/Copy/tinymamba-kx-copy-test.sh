export CUDA_VISIBLE_DEVICES=$1

num_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
platform=langchao
id=$2

model_path=/public/home/ljt/tzc/ckpt/Copy-tinymamba-k${id}-lr1e-3/version_0/checkpoints/last.ckpt

input_seq_len_list="64 128 
                    256 288 320 352 384 416 448 \
                    480 512 544 576 608 640 672 \ 
                    704 736 768 800 832 864 896 \
                    928 960 992 1024"

for input_len in $input_seq_len_list
do
    mark=Copy-tinymamba-k${id}-len${input_len}
    python src/test_dev2.py \
        -mn tiny_mamba-k${id} \
        --ckpt_path $model_path \
        -en $mark \
        -pn $platform \
        -dn Copy \
        --state eval \
        --version 0 \
        --inference_mode \
        --max_seq_len $input_len \
        --processed_data_path Copy/test_${input_len}.pkl;
done

