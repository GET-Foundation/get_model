#!/bin/bash
# Set the path to save checkpoints
DATE=`date +%Y%m%d`
OUTPUT_DIR="/pmglocal/xf2217/Expression_Finetune_k562.Chr4&14.conv20.chrombpnet.shift10.R100L2114.augmented.${DATE}/"
# path to expression set
DATA_PATH='/pmglocal/xf2217/get_data/'
PORT=7957

export NCCL_P2P_LEVEL=NVL
#"/burg/pmg/users/xf2217/get_data/checkpoint-399.from_shentong.R200L500.pth" \

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 --rdzv-endpoint=localhost:$PORT get_model/finetune.py \
    --data_set "Expression_Finetune_THP1.Chr4&14" \
    --eval_data_set "Expression_Finetune_THP1.Chr4&14.Eval" \
    --data_path ${DATA_PATH} \
    --input_dim 639 \
    --output_dim 2 \
    --num_motif 637 \
    --model get_finetune_motif_chrombpnet \
    --batch_size 128 \
    --num_workers 32 \
    --n_peaks_lower_bound 1 \
    --n_peaks_upper_bound 5 \
    --center_expand_target 2114 \
    --preload_count 200 \
    --random_shift_peak 100 \
    --pin_mem \
    --peak_name "peaks_q0.01_tissue_open_exp" \
    --save_ckpt_freq 5 \
    --n_packs 1 \
    --lr 1e-4 \
    --opt adamw \
    --wandb_project_name "chrombpnet" \
    --wandb_run_name "Expression_Finetune_THP1.Chr4&14.conv20.chrombpnet."${DATE} \
    --eval_freq 2 \
    --dist_eval \
    --eval_tss \
    --leave_out_celltypes "THP1" \
    --leave_out_chromosomes "chr4,chr14" \
    --criterion "poisson" \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 1 \
    --epochs 100 \
    --num_region_per_sample 5 \
    --output_dir ${OUTPUT_DIR}
