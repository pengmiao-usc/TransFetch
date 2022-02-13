#!/bin/bash

ChampSimTrace_ROOT="/data/pengmiao/ML-DPC-S0/ChampSimTraces"
LoadTrace_ROOT="/data/pengmiao/ML-DPC-S0/LoadTraces"
ROOT="/home/pengmiao/Project/Intel_2021/MLSYS2022"
#ROOT="/home/pengmiao/Disk/work/HPCA"
OUTPUT_ROOT="/data/pengmiao/CF22/ablation_debug/cx"

Python_ROOT=$ROOT"/1_transformer/ablation_debug/3_context/3_cx"

VERSION=1_trans_128_cx


TRAIN=6
VAL=4
TEST=5
SKIP=1

#TRAIN=5
#VAL=1
#TEST=1
#SKIP=1

TRAIN_WARM=$TRAIN
TRAIN_TOTAL=$(($TRAIN + $VAL)) 

#TEST_WARM=$TRAIN_TOTAL
#TEST_TOTAL=$(($TRAIN + $VAL+$TEST)) 

TEST_WARM=$TRAIN_WARM
TEST_TOTAL=$(($TRAIN+$TEST)) 

app_list=(410.bwaves-s0.txt.xz 429.mcf-s0.txt.xz 433.milc-s0.txt.xz 437.leslie3d-s0.txt.xz 450.soplex-s0.txt.xz 459.GemsFDTD-s0.txt.xz 462.libquantum-s0.txt.xz 470.lbm-s0.txt.xz 471.omnetpp-s0.txt.xz 473.astar-s0.txt.xz 482.sphinx3-s0.txt.xz 602.gcc-s0.txt.xz 605.mcf-s0.txt.xz 607.cactuBSSN-s0.txt.xz 619.lbm-s0.txt.xz 620.omnetpp-s0.txt.xz 621.wrf-s2.txt.xz 623.xalancbmk-s0.txt.xz 649.fotonik3d-s0.txt.xz 654.roms-s0.txt.xz bc-3.txt.xz bfs-3.txt.xz cc-13.txt.xz pr-3.txt.xz sssp-3.txt.xz)

#app_list=(450.soplex-s0.txt.xz 459.GemsFDTD-s0.txt.xz  462.libquantum-s0.txt.xz 470.lbm-s0.txt.xz 471.omnetpp-s0.txt.xz 473.astar-s0.txt.xz 482.sphinx3-s0.txt.xz 602.gcc-s0.txt.xz 605.mcf-s0.txt.xz 607.cactuBSSN-s0.txt.xz 619.lbm-s0.txt.xz 620.omnetpp-s0.txt.xz 621.wrf-s2.txt.xz 623.xalancbmk-s0.txt.xz 649.fotonik3d-s0.txt.xz 654.roms-s0.txt.xz bc-3.txt.xz bfs-3.txt.xz cc-13.txt.xz pr-3.txt.xz sssp-3.txt.xz)


echo "TRAIN/VAL/TEST/SKIP: "$TRAIN"/"$VAL"/"$TEST"/"$SKIP

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

