#!/usr/bin/env bash
source scripts/data_path.sh

THISNAME="blend/geomvsnet"
BESTEPOCH="15"

LOG_DIR="./checkpoints/"$THISNAME
CKPT_FILE=$LOG_DIR"/model_"$BESTEPOCH".ckpt"
TNT_OUT_DIR="./outputs/tnt/"$THISNAME

# Intermediate
CUDA_VISIBLE_DEVICES=0 python3 test.py ${@} \
    --which_dataset="tnt" --loadckpt=$CKPT_FILE --batch_size=1 \
    --outdir=$TNT_OUT_DIR --logdir=$LOG_DIR --nolog \
    --testpath=$TNT_ROOT --testlist="datasets/lists/tnt/intermediate.txt" --split="intermediate" \
    \
    --n_views="11" --img_mode="resize" --cam_mode="origin"

# Advanced
CUDA_VISIBLE_DEVICES=0 python3 test.py ${@} \
    --which_dataset="tnt" --loadckpt=$CKPT_FILE --batch_size=1 \
    --outdir=$TNT_OUT_DIR --logdir=$LOG_DIR --nolog \
    --testpath=$TNT_ROOT --testlist="datasets/lists/tnt/advanced.txt" --split="advanced" \
    \
    --n_views="11" --img_mode="resize" --cam_mode="origin"