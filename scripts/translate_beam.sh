#!/usr/bin/env bash
#!/usr/bin/env bash
res_dir=$2
split_name=$1
checkpoint=$3
python src/translate.py \
--res_dir=${res_dir} \
--eval_splits=${split_name} \
--checkpoint=${checkpoint} \
--use_beam \
--beam_size 2 \
${@:3}
