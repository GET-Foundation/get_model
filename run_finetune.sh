# Set the path to save checkpoints
OUTPUT_DIR='output_e1600_r200_finetune_k562_cut/'
# path to expression set
DATA_PATH='/home/xf2217/Projects/pretrain_human_bingren_shendure_apr2023/'
PORT=7956

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 torchrun --nproc_per_node=1  finetune.py \
    --data_set "Expression" \
    --mask_tss \
    --data_path ${DATA_PATH} \
    --input_dim 283 \
    --criterion "poisson" \
    --data_type k562_cut \
    --model get_finetune_motif \
    --batch_size 2 \
    --leave_out_celltypes "k562_cut0.03,k562_cut0.04,k562_cut0.07" \
    --leave_out_chromosomes "chr1,chr8,chr21" \
    --lr 1.5e-4 \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --use_seq \
    --warmup_epochs 40 \
    --epochs 100 \
    --num_region_per_sample 200 \
    --output_dir ${OUTPUT_DIR} 