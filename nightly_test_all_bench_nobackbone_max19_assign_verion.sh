#!/bin/bash
# $1 gpuid  $2 workdir  $3 date $4 non_ver $5 dis_ver

if [ $# -ne 5 ];
then
    echo "$1 gpuid param required"
    echo "$2 change workdir required"
    echo "S3 date => 20210409-090501"
    echo "$4 non_ver => 41231"
    echo "$5 dis_ver => 32319"
    exit
fi

pushd $2
gpuid=$1
####### define version number #####
# NOW_DATE="20210409-090501"
# NON_CHECKPOINT_NAME="model.ckpt-41231"
# DIS_CHECKPOINT_NAME="model.ckpt-41231"
NOW_DATE=$3
NON_CHECKPOINT_NAME="model.ckpt-$4"
DIS_CHECKPOINT_NAME="model.ckpt-$5"


MODLE_DIT_NAME="bi_lstm_bench_novse_nobackbone_re48_max19"
ADD_PREFIX="re48_max19_"
test_ckpt_dir=model/${MODLE_DIT_NAME}/night_test_ckpt/${NOW_DATE}
NON_CHECKPOINT_DIR=${test_ckpt_dir}/nondisjoint/${NON_CHECKPOINT_NAME}
DIS_CHECKPOINT_DIR=${test_ckpt_dir}/disjoint/${DIS_CHECKPOINT_NAME}


# log name
NON_PARAM="${1} nondisjoint ${NON_CHECKPOINT_DIR} ${ADD_PREFIX}nobackbone_${NOW_DATE}"
DIS_PARAM="${1} disjoint ${DIS_CHECKPOINT_DIR} ${ADD_PREFIX}nobackbone_${NOW_DATE}"

# echo ${test_ckpt_dir}/nondisjoint
# echo ${test_ckpt_dir}/disjoint
# echo ${NON_CHECKPOINT_DIR}
# echo ${DIS_CHECKPOINT_DIR}
echo "====>>  ${NON_PARAM}"
echo "====>>  ${DIS_PARAM}"

echo "=====> begin testing... <======"
echo "extract_feature_bench"
./extract_feature_bench_max19.sh ${NON_PARAM} &
./extract_feature_bench_max19.sh ${DIS_PARAM}

echo "fill_in_blank_bench"
./fill_in_blank_bench_max19.sh ${NON_PARAM} &
./fill_in_blank_bench_max19.sh ${DIS_PARAM}

echo "top_retrieval_bench"
./top_retrieval_bench_max19.sh ${NON_PARAM} &
./top_retrieval_bench_max19.sh ${DIS_PARAM}

echo "predict_compatibility_bench"
./predict_compatibility_bench_max19.sh ${NON_PARAM} &
./predict_compatibility_bench_max19.sh ${DIS_PARAM} &
