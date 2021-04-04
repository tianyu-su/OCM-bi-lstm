#!/bin/bash
# $1 gpuid  $2 which_split  $3 model_path

if [ $# -eq 0 ];
then
    echo "gpuid param required"
    exit
fi

gpuid=$1
NOW_DATE=$(date "+%Y%m%d-%H%M%S")

echo "=====> fetch latest checkpoint version number <======"
REMOTE_DIR="yaoguangan@yga.tb.xpick.cn:/home/share/yaoguangan/benpao/buchongle/baselines/2017MMbilstm/model/bi_lstm_bench_novse"
scp ${REMOTE_DIR}/nondisjoint/train/checkpoint tmp_non_checkpoint
scp ${REMOTE_DIR}/disjoint/train/checkpoint tmp_dis_checkpoint
NON_CHECKPOINT_NAME=`echo $(head -n 1  tmp_non_checkpoint | awk -F '[:]' '{print $NF}') | sed 's/.\(.*\)/\1/' | sed 's/\(.*\)./\1/'`
DIS_CHECKPOINT_NAME=`echo $(head -n 1  tmp_dis_checkpoint | awk -F '[:]' '{print $NF}') | sed 's/.\(.*\)/\1/' | sed 's/\(.*\)./\1/'`
rm tmp_non_checkpoint tmp_dis_checkpoint
echo "nondisjoint VER: ${NON_CHECKPOINT_NAME}"
echo "disjoint VER: ${DIS_CHECKPOINT_NAME}"

####### define version number #####
# NON_CHECKPOINT_NAME="model.ckpt-75650"
# DIS_CHECKPOINT_NAME="model.ckpt-23226"

echo "=====> fetch latest checkpoint bin <======"
test_ckpt_dir=model/bi_lstm_bench_novse/night_test_ckpt/${NOW_DATE}

mkdir -p ${test_ckpt_dir}/nondisjoint
mkdir -p ${test_ckpt_dir}/disjoint

NON_CHECKPOINT_DIR=${test_ckpt_dir}/nondisjoint/${NON_CHECKPOINT_NAME}
DIS_CHECKPOINT_DIR=${test_ckpt_dir}/disjoint/${DIS_CHECKPOINT_NAME}

scp ${REMOTE_DIR}/nondisjoint/train/${NON_CHECKPOINT_NAME}* ${test_ckpt_dir}/nondisjoint/
scp ${REMOTE_DIR}/disjoint/train/${DIS_CHECKPOINT_NAME}* ${test_ckpt_dir}/disjoint

NON_PARAM="${1} nondisjoint ${NON_CHECKPOINT_DIR} ${NOW_DATE}"
DIS_PARAM="${1} disjoint ${DIS_CHECKPOINT_DIR} ${NOW_DATE}"

# echo ${test_ckpt_dir}/nondisjoint
# echo ${test_ckpt_dir}/disjoint
# echo ${NON_CHECKPOINT_DIR}
# echo ${DIS_CHECKPOINT_DIR}
echo "====>>  ${NON_PARAM}"
echo "====>>  ${DIS_PARAM}"

echo "=====> begin testing... <======"
echo "extract_feature_bench"
./extract_feature_bench.sh ${NON_PARAM} &
./extract_feature_bench.sh ${DIS_PARAM}

echo "fill_in_blank_bench"
./fill_in_blank_bench.sh ${NON_PARAM} &
./fill_in_blank_bench.sh ${DIS_PARAM}

echo "top_retrieval_bench"
./top_retrieval_bench.sh ${NON_PARAM} &
./top_retrieval_bench.sh ${DIS_PARAM}

echo "predict_compatibility_bench"
./predict_compatibility_bench.sh ${NON_PARAM} &
./predict_compatibility_bench.sh ${DIS_PARAM} &
