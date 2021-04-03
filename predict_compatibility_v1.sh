#!/bin/bash
CHECKPOINT_DIR="model/bi_lstm/train/model.ckpt-37967"


CUDA_VISIBLE_DEVICES=1 python -u polyvore/fashion_compatibility.py \
  --checkpoint_path=${CHECKPOINT_DIR} \
  --label_file="data/label/fashion_compatibility_prediction.txt" \
  --feature_file="data/features/test_features.pkl" \
  --rnn_type="lstm" \
  --direction="2" \
  --result_file="fashion_compatibility.pkl" > local_res/vse_cp.log 2>&1





CHECKPOINT_DIR="model/bi_lstm_novse/train/model.ckpt-34142"
CUDA_VISIBLE_DEVICES=1 python -u polyvore/fashion_compatibility.py \
  --checkpoint_path=${CHECKPOINT_DIR} \
  --label_file="data/label/fashion_compatibility_prediction.txt" \
  --feature_file="data/features/test_features_novse.pkl" \
  --rnn_type="lstm" \
  --direction="2" \
  --result_file="fashion_compatibility_novse.pkl" > local_res/novse_cp.log 2>&1









