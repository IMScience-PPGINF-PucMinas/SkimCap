#!/usr/bin/env bash
#!/usr/bin/env bash
res_dir=$1
split_name=$2
python src/translate.py \
--res_dir=${res_dir} \
--eval_splits=${split_name} \
--use_beam \
--beam_size 2 \
${@:3}
