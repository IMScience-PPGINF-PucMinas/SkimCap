#!/bin/bash
# Wrapper script for running HieTaSumm with command line arguments

if [ "$#" -ne 4 ]; then
    echo "Usage: bash generate_features.sh [features_path] [summary_path] [gen_summary_method] [hierarchy]"
    exit 1
fi

FEATURES_PATH=$1
SUMMARY_PATH=$2
GEN_SUMMARY_METHOD=$3
HIERARCHY=$4

# Run the Python module with the provided arguments
python -m HieTaSumm.One_Line_Function "$FEATURES_PATH" "$SUMMARY_PATH" "$GEN_SUMMARY_METHOD" "$HIERARCHY"
