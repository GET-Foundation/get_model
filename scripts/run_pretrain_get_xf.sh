#!/bin/bash
# Set the path to save checkpoints
OUTPUT_DIR='/pmglocal/xf2217/20240204.pretrain_conv50_depth4096_500_region_200bp/'
# path to expression set
DATA_PATH='/pmglocal/xf2217/get_data/'
PORT=7956

export NCCL_P2P_LEVEL=NVL

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 torchrun --nproc_per_node=6 --rdzv-endpoint=localhost:$PORT get_model/pretrain.py \
    --data_set "Pretrain" \
    --eval_data_set "Pretrain.GBM_eval" \
    --data_path ${DATA_PATH} \
    --input_dim 639 \
    --output_dim 800 \
    --num_motif 637 \
    --mask_ratio 0.5 \
    --model get_pretrain_motif_base \
    --batch_size 16 \
    --num_workers 64 \
    --n_peaks_lower_bound 20 \
    --n_peaks_upper_bound 500 \
    --center_expand_target 200 \
    --preload_count 200 \
    --pin_mem \
    --eval_nonzero \
    --peak_name "peaks_q0.01_tissue_open" \
    --n_packs 1 \
    --lr 1e-3 \
    --opt adamw \
    --wandb_project_name "get_pretrain" \
    --wandb_run_name "20240204.pretrain_conv50_depth4096_500_region_200bp" \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 10  \
    --epochs 500 \
    --num_region_per_sample 500 \
    --output_dir ${OUTPUT_DIR} 
