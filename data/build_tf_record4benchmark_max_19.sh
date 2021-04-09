#!/bin/bash

preprocessing() {
WHICH_SPLIT=$1
tf_records=data/tf_records/max_19/${WHICH_SPLIT}/
mkdir ${tf_records}

CUDA_VISIBLE_DEVICES=1 python -u data/benchmark_dataset/build_polyvore_data_benchmark_max_19.py \
    --train_label data/benchmark_dataset/label_19/${WHICH_SPLIT}/train_no_dup.json \
    --test_label data/benchmark_dataset/label_19/${WHICH_SPLIT}/test_no_dup.json \
    --valid_label data/benchmark_dataset/label_19/${WHICH_SPLIT}/valid_no_dup.json \
    --output_directory ${tf_records} \
    --image_dir data/polyvore_outfits/images/ \
    --word_dict_file data/final_word_dict.txt
}

preprocessing "nondisjoint"
preprocessing "disjoint"
