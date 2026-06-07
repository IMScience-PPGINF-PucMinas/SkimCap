#!/usr/bin/env bash
# Usage:
# $ bash scripts/build_vocab_clip.sh anet

python src/build_vocab_clip.py \
    --lang_feature_dir video_feature/lang_feature \
    --output_path      cache/anet_vocab_clip.pt \
    --clip_model       ViT-B/16


