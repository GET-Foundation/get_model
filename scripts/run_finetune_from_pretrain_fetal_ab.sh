#!/bin/bash

PRETRAIN_NAME="20240204-pretrain_conv50_depth4096_500_region_200bp"
FINETUNE_EXP_NAME="$PRETRAIN_NAME-fetal-leaveout-Astrocyte-chr11-atpm-0.1"
INPUT_CKPT="/pmglocal/alb2281/get_ckpts/input/$PRETRAIN_NAME-checkpoint-170.pth"
OUTPUT_DIR="/pmglocal/alb2281/get_ckpts/output/finetune-from-pretrain-fetal/$FINETUNE_EXP_NAME/"
DATA_PATH="/pmglocal/alb2281/get_data/shendure_final_zarr"
PORT=7962

export NCCL_P2P_LEVEL=NVL

# batch_size can be adjusted according to the graphics card
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 python -m torch.distributed.run --nproc_per_node=2 --rdzv-endpoint=localhost:$PORT /pmglocal/alb2281/repos/get_model/get_model/finetune.py \
    --data_set "Expression_Finetune_Fetal" \
    --eval_data_set "Expression_Finetune_Fetal.fetal_eval" \
    --finetune ${INPUT_CKPT} \
    --data_path ${DATA_PATH} \
    --input_dim 639 \
    --output_dim 2 \
    --num_motif 637 \
    --model get_finetune_motif \
    --batch_size 16 \
    --num_workers 32 \
    --n_peaks_lower_bound 20 \
    --n_peaks_upper_bound 500 \
    --center_expand_target 200 \
    --non_redundant 'depth_4096' \
    --preload_count 200 \
    --pin_mem \
    --peak_name "peaks_q0.01_tissue_open_exp" \
    --n_packs 1 \
    --lr 5e-4 \
    --opt adamw \
    --wandb_project_name "get-finetune-fetal" \
    --wandb_run_name "$FINETUNE_EXP_NAME" \
    --eval_freq 1 \
    --freeze_atac_attention \
    --dist_eval \
    --eval_nonzero \
    --leave_out_celltypes "Astrocyte" \
    --leave_out_chromosomes "chr11" \
    --criterion "poisson" \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 20 \
    --epochs 100 \
    --num_region_per_sample 500 \
    --output_dir ${OUTPUT_DIR} 
