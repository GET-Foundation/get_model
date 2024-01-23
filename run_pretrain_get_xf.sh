#!/bin/bash
# Set the path to save checkpoints
OUTPUT_DIR='/pmglocal/xf2217/output_rev_pretrain_ATACSplitPool_norm_bidirectional_no_insulation/'
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
    --batch_size 32 \
    --num_workers 64 \
    --n_peaks_lower_bound 20 \
    --n_peaks_upper_bound 100 \
    --preload_count 200 \
    --pin_mem \
    --eval_nonzero \
    --peak_name "peaks_q0.01_tissue_open" \
    --n_packs 1 \
    --lr 1e-3 \
    --opt adamw \
    --wandb_project_name "get_pretrain" \
    --wandb_run_name "get_pretrain_bidirection_no_insulation_100_peaks" \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 1 \
    --epochs 100 \
    --num_region_per_sample 100 \
    --output_dir ${OUTPUT_DIR} 
