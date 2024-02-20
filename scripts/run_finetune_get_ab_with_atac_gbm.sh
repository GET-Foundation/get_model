#!/bin/bash

PRETRAIN_NAME="checkpoint-399.from_shentong.R200L500"
FINETUNE_EXP_NAME="$PRETRAIN_NAME-with-atac-loss-GBM-leaveout-Tumor.htan_gbm.C3N-01814_CPT0167860015-chr11"
INPUT_CKPT="/pmglocal/alb2281/get_ckpts/input/$PRETRAIN_NAME.pth"
OUTPUT_DIR="/pmglocal/alb2281/get_ckpts/output/finetune-from-pretrain/$FINETUNE_EXP_NAME/"
DATA_PATH="/pmglocal/alb2281/get_data"
PORT=7960

export NCCL_P2P_LEVEL=NVL


# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=8 --rdzv-endpoint=localhost:$PORT /pmglocal/alb2281/repos/get_model/get_model/finetune.py \
    --data_set "HTAN_GBM" \
    --eval_data_set "HTAN_GBM.eval" \
    --finetune ${INPUT_CKPT} \
    --data_path ${DATA_PATH} \
    --input_dim 639 \
    --output_dim 2 \
    --num_motif 637 \
    --model get_finetune_motif_with_atac \
    --batch_size 32 \
    --num_workers 32 \
    --n_peaks_lower_bound 20 \
    --n_peaks_upper_bound 100 \
    --center_expand_target 1000 \
    --preload_count 200 \
    --pin_mem \
    --peak_name "peaks_q0.01_tissue_open_exp" \
    --n_packs 1 \
    --lr 5e-4 \
    --opt adamw \
    --wandb_project_name "get-finetune" \
    --wandb_run_name "$FINETUNE_EXP_NAME" \
    --eval_freq 1 \
    --dist_eval \
    --eval_tss \
    --leave_out_celltypes "Tumor.htan_gbm.C3N-01814_CPT0167860015_snATAC_GBM_Tumor" \
    --leave_out_chromosomes "chr4,chr14" \
    --criterion "poisson" \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 20 \
    --epochs 100 \
    --num_region_per_sample 100 \
    --output_dir ${OUTPUT_DIR}
