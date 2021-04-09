#!/bin/bash
# param $1 which $2 gpu_id
WHICH_SPLIT=$1

# Inception v3 checkpoint file.
INCEPTION_CHECKPOINT="model/inception_v3.ckpt"

# Directory to save the model.
# MODEL_DIR="model/test/${WHICH_SPLIT}"
MODEL_DIR="model/bi_lstm_bench_novse_nobackbone_re48_max19/${WHICH_SPLIT}"

# Run the training code.
CUDA_VISIBLE_DEVICES=$2 python -u polyvore/train_bench.py \
  --input_file_pattern="data/tf_records/max_19/${WHICH_SPLIT}/train-no-dup-?????-of-00128" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=false \
  --number_of_steps=100000 \
  --max_outfit_length=19 \
  --batch_size=10 \
  --emb_loss_factor=0.0 > local_logs/re48_max_19_no_backbone_bench_${WHICH_SPLIT}_novse.log 2>&1
#   --emb_loss_factor=0.0


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