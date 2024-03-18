#!/bin/bash
# Set the path to save checkpoints
DATE=`date +%Y%m%d`
OUTPUT_DIR="/pmglocal/xf2217/Expression_Finetune_THP1.mll.Chr4&14.conv20.chrombpnet_withbias.revcomp.bg.peakset.new.${DATE}"
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
    --n_peaks_upper_bound 1 \
    --center_expand_target 2014 \
    --preload_count 200 \
    --random_shift_peak 500 \
    --pin_mem \
    --peak_name "peaks_p0.01_tissue_open_exp" \
    --negative_peak_name "peaks_negative_tissue_open" \
    --negative_peak_ratio 0.1 \
    --save_ckpt_freq 1 \
    --n_packs 1 \
    --lr 1e-3 \
    --opt adamw \
    --wandb_project_name "chrombpnet" \
    --wandb_run_name "Expression_Finetune_THP1.Chr4&14.mnll.conv20.chrombpnet."${DATE} \
    --eval_freq 1 \
    --dist_eval \
    --eval_tss \
    --leave_out_celltypes "THP1" \
    --leave_out_chromosomes "chr3,chr4,chr13,chr14" \
    --criterion "poisson" \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 1 \
    --epochs 50 \
    --num_region_per_sample 1 \
    --output_dir ${OUTPUT_DIR}
