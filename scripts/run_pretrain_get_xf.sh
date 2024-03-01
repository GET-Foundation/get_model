#!/bin/bash
# Set the path to save checkpoints
DATE=`date +%Y%m%d`
OUTPUT_DIR='/pmglocal/xf2217/pretrain_conv50.maxnorm.R100L500.${DATE}'
# path to expression set
DATA_PATH='/pmglocal/xf2217/get_data/'

PORT=7956

export NCCL_P2P_LEVEL=NVL

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 --rdzv-endpoint=localhost:$PORT get_model/pretrain.py \
    --data_set "Pretrain" \
    --eval_data_set "Pretrain.GBM_eval" \
    --data_path ${DATA_PATH} \
    --input_dim 639 \
    --output_dim 800 \
    --num_motif 637 \
    --mask_ratio 0.5 \
    --model get_pretrain_motif_base_maxnorm \
    --batch_size 32 \
    --num_workers 32 \
    --use_insulation \
    --n_peaks_lower_bound 10 \
    --n_peaks_upper_bound 100 \
    --center_expand_target 500 \
    --preload_count 200 \
    --random_shift_peak \
    --pin_mem \
    --eval_nonzero \
    --peak_name "peaks_q0.01_tissue_open" \
    --n_packs 1 \
    --lr 1e-3 \
    --opt adamw \
    --wandb_project_name "get_pretrain" \
    --wandb_run_name "pretrain_conv50.R100L500.${DATE}" \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 10  \
    --epochs 600 \
    --num_region_per_sample 100 \
    --output_dir ${OUTPUT_DIR} 

