#!/usr/bin/env bash
# Usage:
#   $ bash {SCRIPT.sh} [Any flags available in train.py, could also be empty]
# Examples:
#   anet debug mode:    $ bash scripts/train.sh --debug
#   anet training mode: $ bash scripts/train.sh

dset_name="anet"

data_dir="./densevid_eval/${dset_name}_data"
v_feat_dir="./video_feature/cd_anet_feat"
flow_feat_dir="./video_feature/rt_anet_feat/trainval"
dur_file="./video_feature/anet_duration_frame.csv"
word2idx_path="./cache/${dset_name}_word2idx.json"
glove_path="./cache/${dset_name}_vocab_glove.pt"

echo "---------------------------------------------------------"
echo ">>>>>>>> Running training on ${dset_name} dataset"

if [[ ${dset_name} == "anet" ]]; then
    max_n_sen=6
    max_t_len=22
    max_v_len=100
elif [[ ${dset_name} == "yc2" ]]; then
    max_n_sen=12
    max_t_len=22
    max_v_len=100
else
    echo "Wrong dataset name: select between anet and yc2"
    exit 1
fi

time python src/train.py \
    --dset_name ${dset_name} \
    --data_dir ${data_dir} \
    --video_feature_dir ${v_feat_dir} \
    --flow_feature_dir ${flow_feat_dir} \
    --v_duration_file ${dur_file} \
    --word2idx_path ${word2idx_path} \
    --glove_path ${glove_path} \
    --max_n_sen ${max_n_sen} \
    --max_t_len ${max_t_len} \
    --max_v_len ${max_v_len} \
    --video_feature_size 3072 \
    --n_epoch 50 \
    --exp_id init \
    --batch_size 160 \
    --num_workers 8 \
    --n_memory_cells 1 \
    --intermediate_size 768 \
    --hidden_size 768 \
    --recurrent \
    "$@"