#!/bin/bash
mamba activate geneformer
# Set the path to save checkpoints
OUTPUT_DIR='/home/xf2217/Projects/all_finetune/output'
# path to expression set
DATA_PATH='/home/xf2217/Projects/all_finetune/data/'
PORT=7956

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 finetune.py \
    --data_set "Expression" \
    --mask_tss \
    --data_path ${DATA_PATH} \
    --input_dim 283 \
    --eval_freq 5 \
    --criterion "poisson" \
    --data_type ball \
    --model get_finetune_motif \
    --use_natac \
    --batch_size 16 \
    --leave_out_celltypes "795" \
    --leave_out_chromosomes "chr11" \
    --lr 1e-3 \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --epochs 100 \
    --num_region_per_sample 200 \
    --output_dir ${OUTPUT_DIR} 
