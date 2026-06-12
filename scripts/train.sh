#!/usr/bin/env bash

dset_name="anet"
data_dir="/home/lvcardoso/SkimCap/densevid_eval/${dset_name}_data"
v_feat_dir="./video_feature/cd_anet_feat"
resnet_feat_dir="./video_feature/rt_anet_feat/trainval"  # used only with --appearance_feat resnet
dur_file="./video_feature/anet_duration_frame.csv"
word2idx_path="./cache/${dset_name}_word2idx.json"
glove_path="./cache/${dset_name}_vocab_glove.pt"
lang_feat_dir="./video_feature/lang_feature"
sent_feat_dir="./video_feature/sent_feature"
flow_feat_dir="./video_feature/rt_anet_feat/trainval"
clip_feat_dir="./cache/${dset_name}_vocab_clip.pt"
appearance_feat_type="c3d"
batch=96
val_batch=${batch}   # pode ser reduzido se beam_size for aumentado

echo "---------------------------------------------------------"
echo ">>>>>>>> Running training on ${dset_name} dataset (CLIP + batch ${batch})"

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
    --flow_feature_dir ${flow_feat_dir} \
    --v_duration_file ${dur_file} \
    --word2idx_path ${word2idx_path} \
    --glove_path ${glove_path} \
    --lang_feature_dir ${lang_feat_dir} \
    --sent_feature_dir ${sent_feat_dir} \
    --vocab_clip_path ${clip_feat_dir} \
    --appearance_feat ${appearance_feat_type} \
    --max_n_sen ${max_n_sen} \
    --max_t_len ${max_t_len} \
    --max_v_len ${max_v_len} \
    --video_feature_size 3072  \
    --lang_feature_size 512 \
    --n_epoch 50 \
    --use_beam \
    --beam_size 2 \
    --lr 1.5e-4 \
    --lr_warmup_proportion 0.1 \
    --label_smoothing 0.05 \
    --contrastive_temp 0.10 \
    --contrastive_weight 0.1 \
    --sent_loss_weight 0.15 \
    --batch_size ${batch} \
    --val_batch_size ${val_batch} \
    --max_es_cnt 15 \
    --num_workers 0 \
    --n_memory_cells 8 \
    --num_hidden_layers 4 \
    --intermediate_size 768 \
    --hidden_size 768 \
    --num_attention_heads 12 \
    --ema_decay 0.9996 \
    --recurrent \
    --exp_id ${appearance_feat_type}_clip_b${batch}_no_flow \
    --no_flow \
    "$@"

# ── Ablation examples ────────────────────────────────────────────────────────
# Adicione os flags abaixo ao comando acima para isolar a contribuição de cada módulo.
#
#  --appearance_feat c3d                     backbone C3D (padrão, 100 clips fixos)
#  --appearance_feat resnet \               backbone ResNet-200 (variável, auto-resample)
#      --resnet_feature_dir ${resnet_feat_dir}   obrigatório com resnet
#
#  --no_flow               sem optical flow (apenas aparência visual)
#  --no_lang               sem CLIP lang features (por frame)
#  --no_sent               sem CLIP sent features (alinhamento semântico)
#  --no_lang --no_sent     baseline sem CLIP
#  --no_flow --no_lang --no_sent   equivalente ao MART original
#
# video_feature_size é calculado automaticamente — não precisa ser alterado.
# ─────────────────────────────────────────────────────────────────────────────