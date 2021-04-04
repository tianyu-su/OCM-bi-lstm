#!/bin/bash
# $1 gpuid  $2 which_split  $3 model_path

# gpuid=$1
# WHICH_SPLIT=$2
#
# pre_fix=$4_
# if [ ! -n "$4"  ];
# then
#     pre_fix=""
# fi
#
# echo ${pre_fix}12


#
# aa=`echo "model_checkpoint_path: "model.ckpt-76316"" | awk -F '[:]' '{print $NF}'`
# echo ${aa}


file=\"model.ckpt-77641\"
echo ${file}
temp=`echo $file | sed 's/.\(.*\)/\1/' | sed 's/\(.*\)./\1/'`
echo ${temp}