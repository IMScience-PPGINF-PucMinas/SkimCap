#!/usr/bin/env bash

dset_name="anet"
data_dir="/home/lvcardoso/SkimCap/densevid_eval/${dset_name}_data"
v_feat_dir="./video_feature/cd_anet_feat"
dur_file="./video_feature/anet_duration_frame.csv"
word2idx_path="./cache/${dset_name}_word2idx.json"
glove_path="./cache/${dset_name}_vocab_glove.pt"
lang_feat_dir="./video_feature/lang_feature"    # ajuste para seu path
sent_feat_dir="./video_feature/sent_feature"    # ajuste para seu path

echo "---------------------------------------------------------"
echo ">>>>>>>> Running training on ${dset_name} dataset (CLIP + batch 128)"

if [[ ${dset_name} == "anet" ]]; then
    max_n_sen=6
    max_t_len=32
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
    --v_duration_file ${dur_file} \
    --word2idx_path ${word2idx_path} \
    --glove_path ${glove_path} \
    --lang_feature_dir ${lang_feat_dir} \
    --sent_feature_dir ${sent_feat_dir} \
    --max_n_sen ${max_n_sen} \
    --max_t_len ${max_t_len} \
    --max_v_len ${max_v_len} \
    --video_feature_size 2048 \
    --lang_feature_size 512 \
    --n_epoch 100 \
    --lr 1.5e-4 \
    --lr_warmup_proportion 0.15 \
    --label_smoothing 0.05 \
    --contrastive_temp 0.10 \
    --contrastive_weight 0.1 \
    --sent_loss_weight 0.15 \
    --batch_size 96 \
    --val_batch_size 64 \
    --max_es_cnt 15 \
    --num_workers 8 \
    --n_memory_cells 8 \
    --num_hidden_layers 4 \
    --intermediate_size 768 \
    --hidden_size 768 \
    --num_attention_heads 12 \
    --ema_decay 0.9996 \
    --recurrent \
    --exp_id clip_b128_woflow \
    "$@"