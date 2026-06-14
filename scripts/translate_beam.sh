#!/usr/bin/env bash
split_name=$1
res_dir=$2
checkpoint=$3
app_feat="resnet"
shift 3   # consume the 3 positional args; "$@" now holds only extra flags

python src/translate.py \
  --res_dir="${res_dir}" \
  --eval_splits="${split_name}" \
  --checkpoint="${checkpoint}" \
  --use_beam \
  --appearance_feat=${app_feat} \
  --beam_size 2 \
  "$@"