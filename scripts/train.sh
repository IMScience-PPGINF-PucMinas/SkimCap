#!/usr/bin/env bash

dset_name="anet"
data_dir="/home/lvcardoso/SkimCap/densevid_eval/${dset_name}_data"
v_feat_dir="./video_feature/cd_anet_feat"
dur_file="./video_feature/anet_duration_frame.csv"
word2idx_path="./cache/${dset_name}_word2idx.json"
glove_path="./cache/${dset_name}_vocab_glove.pt"
lang_feat_dir="./video_feature/lang_feature"
sent_feat_dir="./video_feature/sent_feature"

echo "---------------------------------------------------------"
echo ">>>>>>>> Running training on ${dset_name} dataset (C3D + CLIP lang early-concat)"

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
    --v_duration_file ${dur_file} \
    --word2idx_path ${word2idx_path} \
    --glove_path ${glove_path} \
    --feature_type c3d \
    --lang_feature_size 0 \
    --vocab_clip_path cache/anet_vocab_clip.pt \
    --max_n_sen ${max_n_sen} \
    --max_t_len ${max_t_len} \
    --max_v_len ${max_v_len} \
    --video_feature_size 2048 \
    --contrastive_weight 0.0 \
    --sent_loss_weight 0.25 \
    --n_epoch 50 \
    --use_beam \
    --beam_size 2 \
    --lr 1.5e-4 \
    --lr_warmup_proportion 0.1 \
    --label_smoothing 0.05 \
    --batch_size 96 \
    --val_batch_size 64 \
    --max_es_cnt 10 \
    --num_workers 8 \
    --n_memory_cells 1 \
    --num_hidden_layers 2 \
    --intermediate_size 768 \
    --hidden_size 768 \
    --num_attention_heads 12 \
    --ema_decay 0.9996 \
    --recurrent \
    --exp_id c3d_lang_early_concat_contrastive \
    "$@"