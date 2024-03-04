#!/bin/bash
# ATAC baseline
ATAC_DATA="/pmglocal/alb2281/get_data/k562_count_10/k562_count_10.csv"
LABELS_PATH="/pmglocal/alb2281/get_data/k562_count_10/k562_count_10.watac.npz"
LEAVEOUT_CHR="chr11"
FINETUNE_EXP_NAME="enformer_finetune_atac_leaveout_$LEAVEOUT_CHR"
OUTPUT_DIR="/pmglocal/alb2281/get_ckpts/output/k562-baseline/$FINETUNE_EXP_NAME/"
PORT=7960

export NCCL_P2P_LEVEL=NVL

# batch_size can be adjusted according to the graphics card
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=2 --rdzv-endpoint=localhost:$PORT /pmglocal/alb2281/repos/get_model/baselines/enformer/finetune_enformer.py \
    --atac_data ${DATA_PATH} \
    --labels_path ${LABELS_PATH} \
    --batch_size 16 \
    --num_workers 32 \
    --lr 5e-4 \
    --opt adamw \
    --wandb_project_name "get-baselines-atac" \
    --wandb_run_name "$FINETUNE_EXP_NAME" \
    --eval_freq 1 \
    --dist_eval \
    --leave_out_chromosomes ${LEAVEOUT_CHR} \
    --warmup_epochs 20 \
    --epochs 100 \
    --output_dir ${OUTPUT_DIR} 
