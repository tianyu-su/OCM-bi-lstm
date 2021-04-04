#!/bin/bash
# $1 gpuid  $2 which_split  $3 model_path

gpuid=$1
WHICH_SPLIT=$2
CHECKPOINT_DIR=$3
# CHECKPOINT_DIR="model/bi_lstm_novse/train/model.ckpt-34142"
./extract_feature_bench.sh $1 $2 $3
./fill_in_blank_bench.sh $1 $2 $3
./top_retrieval_bench.sh $1 $2 $3
./predict_compatibility_bench.sh $1 $2 $3