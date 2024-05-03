#!/bin/bash
ATAC_DATA="/pmglocal/alb2281/get/get_rebuttal/data/raw/k562-benchmark/k562_count_10/k562_count_10.csv"
LABELS_PATH="/pmglocal/alb2281/get/get_rebuttal/data/raw/k562-benchmark/k562_count_10/k562_count_10.watac.npz"
LEAVEOUT_CHR="chr10,chr11"
FINETUNE_EXP_NAME="enformer_finetune_atac_leaveout_$LEAVEOUT_CHR"

# batch_size can be adjusted according to the graphics card
CUDA_VISIBLE_DEVICES=1 python /pmglocal/alb2281/get/get_rebuttal/baselines/enformer/finetune_enformer.py \
    --atac_data ${ATAC_DATA} \
    --labels_path ${LABELS_PATH} \
    --batch_size 4 \
    --leave_out_chr ${LEAVEOUT_CHR} 
