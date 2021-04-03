#!/bin/bash
# $1 gpuid  $2 which_split  $3 model_path

gpuid=$1
WHICH_SPLIT=$2
CHECKPOINT_DIR=$3
# CHECKPOINT_DIR="model/bi_lstm/train/model.ckpt-37967"

CUDA_VISIBLE_DEVICES=${gpuid} python -u polyvore/run_inference_bench.py \
  --checkpoint_path=${CHECKPOINT_DIR} \
  --json_file data/benchmark_dataset/label/${WHICH_SPLIT}/test_no_dup.json \
  --image_dir data/polyvore_outfits/images/ \
  --feature_file data/features/${WHICH_SPLIT}/test_features.pkl \
  --rnn_type lstm \
  --setid2id_mapping_file data/benchmark_dataset/label/${WHICH_SPLIT}/set2id_mapping.json

