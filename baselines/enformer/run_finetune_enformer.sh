#!/bin/bash
# ATAC baseline
LEAVEOUT_CHR="chr11"
RUN_NAME="enformer_finetune_atac_leaveout_$LEAVEOUT_CHR"

CUDA_VISIBLE_DEVICES=0 python /pmglocal/alb2281/repos/get_model/baselines/enformer/enformer.py \
    --batch_size 2 \
    --num_workers 1 \
    --num_epochs 10 \
    --accumulation_steps 1 \
    --learning_rate 1e-5 \
    --eval_freq 5 \
    --leaveout_chr ${LEAVEOUT_CHR} \
    --wandb_project_name "get-baselines" \
    --wandb_entity_name "get-v3" \
    --wandb_run_name ${RUN_NAME} 
