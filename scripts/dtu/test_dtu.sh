#!/usr/bin/env bash
source scripts/data_path.sh

THISNAME="geomvsnet"
BESTEPOCH="15"

LOG_DIR="./checkpoints/dtu/"$THISNAME 
DTU_CKPT_FILE=$LOG_DIR"/model_"$BESTEPOCH".ckpt"
DTU_OUT_DIR="./outputs/dtu/"$THISNAME

CUDA_VISIBLE_DEVICES=0 python3 test.py ${@} \
    --which_dataset="dtu" --loadckpt=$DTU_CKPT_FILE --batch_size=1 \
    --outdir=$DTU_OUT_DIR --logdir=$LOG_DIR --nolog \
    --testpath=$DTU_TEST_ROOT --testlist="datasets/lists/dtu/test.txt" \
    \
    --data_scale="raw" --n_views="5"