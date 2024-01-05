#!/bin/bash
# Set the path to save checkpoints
OUTPUT_DIR='/pmglocal/xf2217/output_pretrain_rev/'
# path to expression set
DATA_PATH='/pmglocal/xf2217/get_data/'
PORT=7956

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 --rdzv-endpoint=localhost:$PORT get_model/pretrain.py \
    --data_set "Pretrain" \
    --data_path ${DATA_PATH} \
    --input_dim 1274 \
    --num_motif 637 \
    --mask_ratio 0.5 \
    --model get_pretrain_motif_base \
    --batch_size 16 \
    --num_workers 64 \
    --preload_count 200 \
    --pin_mem \
    --peak_name "peaks_q0.01_tissue" \
    --n_packs 1 \
    --flash_attn \
    --lr 1e-3 \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 1 \
    --epochs 100 \
    --num_region_per_sample 200 \
    --output_dir ${OUTPUT_DIR} 
