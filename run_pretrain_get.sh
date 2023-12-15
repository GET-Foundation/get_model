#!/bin/bash
# Set the path to save checkpoints
OUTPUT_DIR='./output_pretrain_natac/'
# path to expression set
DATA_PATH='../../../data/pretrain_human_bingren_shendure_apr2023/'
PORT=7956


# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 --rdzv-endpoint=localhost:$PORT get_model/pretrain.py \
    --data_set "Pretrain" \
    --data_path ${DATA_PATH} \
    --input_dim 1274 \
    --num_motif 637 \
    --data_type k562_cut \
    --mask_ratio 0.5 \
    --model get_pretrain_motif_base \
    --batch_size 4 \
    --leave_out_celltypes "k562_cut0.03,k562_cut0.04,k562_cut0.07" \
    --lr 1e-3 \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 40 \
    --use_seq \
    --epochs 1600 \
    --num_region_per_sample 200 \
    --output_dir ${OUTPUT_DIR} 
