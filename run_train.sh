#!/bin/bash

VERSION=1_trans_128_cx

ROOT="./"
LoadTrace_ROOT="./SampleData/LoadTraces"
OUTPUT_ROOT="./res"

Python_ROOT=$ROOT"/TransFetch"

TRAIN=40
VAL=10
TEST=50
SKIP=1

TRAIN_WARM=$TRAIN
TRAIN_TOTAL=$(($TRAIN + $VAL)) 

TEST_WARM=$TRAIN_WARM
TEST_TOTAL=$(($TRAIN+$TEST)) 

app_list=(410.bwaves-s0.txt.xz)


echo "TRAIN/VAL/TEST/SKIP: "$TRAIN"/"$VAL"/"$TEST"/"$SKIP

mkdir $OUTPUT_ROOT
mkdir $OUTPUT_ROOT/$VERSION
mkdir $OUTPUT_ROOT/$VERSION/train

#for app1 in `ls $LoadTrace_ROOT`; do
for app1 in ${app_list[*]}; do
	echo $app1
	file_path=$LoadTrace_ROOT/${app1}
    model_path=$OUTPUT_ROOT/$VERSION/train/${app1}.model.pth
	#app2=${app1%%.txt*}.trace.xz

    python $Python_ROOT/train.py  $file_path  $model_path $TRAIN_WARM $TRAIN_TOTAL $SKIP
    python $Python_ROOT/generate.py  $file_path  $model_path $TEST_WARM $TEST_TOTAL $SKIP
    
	echo "done for app "$app1
done

