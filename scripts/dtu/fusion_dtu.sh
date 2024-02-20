#!/usr/bin/env bash
source scripts/data_path.sh

THISNAME="geomvsnet"
FUSION_METHOD="open3d"

LOG_DIR="./checkpoints/dtu/"$THISNAME 
DTU_OUT_DIR="./outputs/dtu/"$THISNAME

if [ $FUSION_METHOD = "pcd" ] ; then
python3 fusions/dtu/pcd.py ${@} \
    --testpath=$DTU_TEST_ROOT --testlist="datasets/lists/dtu/test.txt" \
    --outdir=$DTU_OUT_DIR --logdir=$LOG_DIR --nolog \
    --num_worker=1 \
    \
    --thres_view=4 --conf=0.5 \
    \
    --plydir=$DTU_OUT_DIR"/pcd_fusion_plys/"
    
elif [ $FUSION_METHOD = "gipuma" ] ; then
# source [/path/to/]anaconda3/etc/profile.d/conda.sh
# conda activate fusibile
CUDA_VISIBLE_DEVICES=0 python2 fusions/dtu/gipuma.py \
    --root_dir=$DTU_TEST_ROOT --list_file="datasets/lists/dtu/test.txt" \
    --fusibile_exe_path="fusions/fusibile" --out_folder="fusibile_fused" \
    --depth_folder=$DTU_OUT_DIR \
    --downsample_factor=1 \
    \
    --prob_threshold=0.5 --disp_threshold=0.25 --num_consistent=3 \
    \
    --plydir=$DTU_OUT_DIR"/gipuma_fusion_plys/"

elif [ $FUSION_METHOD = "open3d" ] ; then
CUDA_VISIBLE_DEVICES=0 python fusions/dtu/_open3d.py --device="cuda" \
    --root_path=$DTU_TEST_ROOT \
    --depth_path=$DTU_OUT_DIR \
    --data_list="datasets/lists/dtu/test.txt" \
    \
    --prob_thresh=0.3 --dist_thresh=0.2 --num_consist=4 \
    \
    --ply_path=$DTU_OUT_DIR"/open3d_fusion_plys/"

fi
