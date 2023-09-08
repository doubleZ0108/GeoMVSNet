#!/usr/bin/env bash
source scripts/data_path.sh

OUTNAME="geomvsnet"

FUSIONMETHOD="open3d"

# Evaluation
echo "<<<<<<<<<< start parallel evaluation"
METHOD='mvsnet'
PLYPATH='../../../outputs/dtu/'$OUTNAME'/'$FUSIONMETHOD'_fusion_plys/'
RESULTPATH='../../../outputs/dtu/'$OUTNAME'/'$FUSIONMETHOD'_quantitative/'
LOGPATH='outputs/dtu/'$OUTNAME'/'$FUSIONMETHOD'_quantitative/'$OUTNAME'.log'

mkdir -p 'outputs/dtu/'$OUTNAME'/'$FUSIONMETHOD'_quantitative/'

set_array=(1 4 9 10 11 12 13 15 23 24 29 32 33 34 48 49 62 75 77 110 114 118)

num_at_once=2   # 1 2 4 5 7 11 22
times=`expr $((${#set_array[*]} / $num_at_once))`
remain=`expr $((${#set_array[*]} - $num_at_once * $times))`
this_group_num=0
pos=0

for ((t=0; t<$times; t++))
do
    if [ "$t" -ge `expr $(($times-$remain))` ] ; then
        this_group_num=`expr $(($num_at_once + 1))`
    else
        this_group_num=$num_at_once
    fi
    
    for set in "${set_array[@]:pos:this_group_num}"
    do
        matlab -nodesktop -nosplash -r "cd datasets/evaluations/dtu_parallel; dataPath='$DTU_QUANTITATIVE_ROOT'; plyPath='$PLYPATH'; resultsPath='$RESULTPATH'; method_string='$METHOD'; thisset='$set'; BaseEvalMain_web" &
    done
    wait

    pos=`expr $(($pos + $this_group_num))`

done
wait


SET=[1,4,9,10,11,12,13,15,23,24,29,32,33,34,48,49,62,75,77,110,114,118]

matlab -nodesktop -nosplash -r "cd datasets/evaluations/dtu_parallel; resultsPath='$RESULTPATH'; method_string='$METHOD'; set='$SET'; ComputeStat_web" > $LOGPATH