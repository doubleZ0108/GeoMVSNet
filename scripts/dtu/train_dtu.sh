#!/usr/bin/env bash
source scripts/data_path.sh

THISNAME="geomvsnet"

LOG_DIR="./checkpoints/dtu/"$THISNAME 
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py ${@} \
    --which_dataset="dtu" --epochs=16 --logdir=$LOG_DIR \
    --trainpath=$DTU_TRAIN_ROOT --testpath=$DTU_TRAIN_ROOT \
    --trainlist="datasets/lists/dtu/train.txt" --testlist="datasets/lists/dtu/test.txt" \
    \
    --data_scale="mid" --n_views="5" --batch_size=4 --lr=0.002 --robust_train \
    --lrepochs="1,3,5,7,9,11,13,15:1.5"