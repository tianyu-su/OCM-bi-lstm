#!/bin/bash
# CHECKPOINT_DIR="model/bi_lstm/train/model.ckpt-37967"
CHECKPOINT_DIR="model/bi_lstm/train/model.ckpt-40468"


CUDA_VISIBLE_DEVICES=0 python -u polyvore/fill_in_blank.py \
  --checkpoint_path=${CHECKPOINT_DIR} \
  --json_file="data/label/fill_in_blank_test.json" \
  --feature_file="data/features/test_features.pkl" \
  --rnn_type="lstm" \
  --direction="2" \
  --result_file="fill_in_blank_result.pkl" > local_res/vse_fitb.log 2>&1





# CHECKPOINT_DIR="model/bi_lstm_novse/train/model.ckpt-34142"
# CUDA_VISIBLE_DEVICES=0 python -u polyvore/fill_in_blank.py \
#   --checkpoint_path=${CHECKPOINT_DIR} \
#   --json_file="data/label/fill_in_blank_test.json" \
#   --feature_file="data/features/test_features_novse.pkl" \
#   --rnn_type="lstm" \
#   --direction="2" \
#   --result_file="fill_in_blank_result_novse.pkl" | tee local_res/novse_fitb.log