#!/usr/bin/env bash
#!/usr/bin/env bash
split_name=$1
res_dir=$2
checkpoint=$3
python src/translate.py \
--res_dir=${res_dir} \
--eval_splits=${split_name} \
--checkpoint=${checkpoint} \
--use_beam \
--beam_size 2 \
"$@"