#!/bin/bash
# Set the path to save checkpoints
OUTPUT_DIR='/pmglocal/xf2217/eval/'
# path to expression set
DATA_PATH='/pmglocal/xf2217/get_data/'
PORT=7957

export NCCL_P2P_LEVEL=NVL

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 --rdzv-endpoint=localhost:$PORT get_model/finetune.py \
    --data_set "Expression_Finetune_Fetal" \
    --eval_data_set "Expression_Finetune_Fetal.fetal_eval" \
    --data_path ${DATA_PATH} \
    --finetune "/burg/pmg/users/xf2217/get_checkpoints/fetal_hsc_gbm.all_chr.best.pth" \
    --input_dim 639 \
    --output_dim 2 \
    --num_motif 637 \
    --eval \
    --model get_finetune_motif_with_atac \
    --batch_size 8 \
    --num_workers 32 \
    --n_peaks_lower_bound 10 \
    --n_peaks_upper_bound 100 \
    --center_expand_target 1000 \
    --preload_count 200 \
    --pin_mem \
    --peak_name "peaks_q0.01_tissue_open_exp" \
    --n_packs 1 \
    --lr 5e-4 \
    --opt adamw \
    --wandb_project_name "get_eval" \
    --wandb_run_name "eval" \
    --eval_freq 5 \
    --freeze_atac_attention \
    --dist_eval \
    --eval \
    --eval_tss \
    --leave_out_celltypes "Astrocyte" \
    --criterion "poisson" \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 20 \
    --epochs 100 \
    --num_region_per_sample 100 \
    --output_dir ${OUTPUT_DIR} 
