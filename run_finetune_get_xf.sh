#!/bin/bash
# Set the path to save checkpoints
OUTPUT_DIR='/pmglocal/xf2217/output_rev_from_scratch_ATACSplitPool_unnorm_finetune_fetal_Erythroblast_leaveout_chr_bidirectional/'
# path to expression set
DATA_PATH='/pmglocal/xf2217/get_data/'
PORT=7956

export NCCL_P2P_LEVEL=NVL

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 --rdzv-endpoint=localhost:$PORT get_model/finetune.py \
    --data_set "Expression_Finetune_Fetal" \
    --eval_data_set "Expression_Finetune_Fetal.fetal_eval" \
    --data_path ${DATA_PATH} \
    --input_dim 639 \
    --output_dim 2 \
    --num_motif 637 \
    --model get_finetune_motif \
    --batch_size 32 \
    --num_workers 64 \
    --n_peaks_lower_bound 10 \
    --n_peaks_upper_bound 100 \
    --preload_count 200 \
    --pin_mem \
    --peak_name "peaks_q0.01_tissue_open_exp" \
    --n_packs 1 \
    --flash_attn \
    --lr 5e-4 \
    --opt adamw \
    --wandb_project_name "get_finetune" \
    --wandb_run_name "ATACSplitPool_finetune_from_scratch_bidirectional" \
    --eval_freq 1 \
    --dist_eval \
    --eval_nonzero \
    --leave_out_celltypes "Astrocyte" \
    --leave_out_chromosomes "chr11" \
    --criterion "poisson" \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 5 \
    --epochs 200 \
    --num_region_per_sample 100 \
    --output_dir ${OUTPUT_DIR} 
