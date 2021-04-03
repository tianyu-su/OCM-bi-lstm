#!/bin/bash

# Inception v3 checkpoint file.
INCEPTION_CHECKPOINT="model/inception_v3.ckpt"

# Directory to save the model.
MODEL_DIR="model/bi_lstm_bench_vse/"

# Run the training code.
CUDA_VISIBLE_DEVICES=2 python -u polyvore/train.py \
  --input_file_pattern="data/tf_records/train-no-dup-?????-of-00128" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=true \
  --number_of_steps=100000 \
  --emb_loss_factor=1.0
  # --emb_loss_factor=1.0 > local_logs/bench_vse.log 2>&1


# # Training Siamese Network
# # Directory to save the model.
# MODEL_DIR="model/siamese/"

# # Run the training code.
# python polyvore/train_siamese.py \
#   --input_file_pattern="data/tf_records/train-no-dup-?????-of-00128" \
#   --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
#   --train_dir="${MODEL_DIR}/train" \
#   --train_inception=true \
#   --number_of_steps=100000