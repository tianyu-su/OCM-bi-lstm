#!/bin/bash
# $1 gpuid  $2 which_split  $3 model_path $4 pre_fix


gpuid=$1
WHICH_SPLIT=$2
CHECKPOINT_DIR=$3
# CHECKPOINT_DIR="model/bi_lstm_novse/train/model.ckpt-34142"

pre_fix=$4_
if [ ! -n "$4"  ];
then
    pre_fix=""
fi
echo "params: ${0} ${1} ${2} ${3} ${4}"

test_logs=test_results/${WHICH_SPLIT}/logs
test_results_file=test_results/${WHICH_SPLIT}

mkdir -p ${test_logs}
CUDA_VISIBLE_DEVICES=${gpuid} python -u polyvore/fill_in_blank.py \
  --checkpoint_path=${CHECKPOINT_DIR} \
  --json_file data/benchmark_dataset/label/${WHICH_SPLIT}/fill_in_blank_test.json \
  --feature_file data/features/${WHICH_SPLIT}/${pre_fix}test_features.pkl \
  --rnn_type="lstm" \
  --direction="2" \
  --result_file ${test_results_file}/${pre_fix}fill_in_blank_result.pkl | tee ${test_logs}/${pre_fix}fitb.log
