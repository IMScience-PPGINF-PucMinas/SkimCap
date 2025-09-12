#!/usr/bin/env bash
res_dir=$1
split_name=$2
python src/translate.py \
--res_dir=${res_dir} \
--eval_splits=${split_name} \
--min_sen_len=6 \
--max_sen_len=32 \
--batch_size=100 \
${@:3}
