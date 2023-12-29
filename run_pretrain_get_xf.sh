#!/bin/bash
# Set the path to save checkpoints
OUTPUT_DIR='/pmglocal/xf2217/output_pretrain_rev/'
# path to expression set
DATA_PATH='/pmglocal/xf2217/get_model/'
PORT=7956


# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 torchrun --nproc_per_node=6 --rdzv-endpoint=localhost:$PORT get_model/pretrain.py \
    --data_set "Pretrain" \
    --data_path ${DATA_PATH} \
    --input_dim 1274 \
    --num_motif 637 \
    --mask_ratio 0.5 \
    --model get_pretrain_motif_base \
    --batch_size 16 \
    --num_workers 16 \
    --lr 1e-3 \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --warmup_steps 1000 \
    --epochs 100 \
    --num_region_per_sample 200 \
    --output_dir ${OUTPUT_DIR} 
