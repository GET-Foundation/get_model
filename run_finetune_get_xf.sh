#!/bin/bash
# Set the path to save checkpoints
OUTPUT_DIR='/pmglocal/xf2217/output_pretrain_rev_ATACSplitPool_unnorm_finetune_fetal_Astrocyte_ood/'
# path to expression set
DATA_PATH='/pmglocal/xf2217/get_data/'
PORT=7956

export NCCL_P2P_LEVEL=NVL

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 --rdzv-endpoint=localhost:$PORT get_model/finetune.py \
    --finetune "/pmglocal/xf2217/output_pretrain_rev_ATACSplitPool_unnorm/checkpoint-2.pth" \
    --data_set "Expression_Finetune_Fetal" \
    --eval_data_set "Expression_Finetune_Fetal.fetal_eval" \
    --data_path ${DATA_PATH} \
    --input_dim 1274 \
    --output_dim 2 \
    --num_motif 637 \
    --model get_finetune_motif \
    --batch_size 16 \
    --num_workers 64 \
    --n_peaks_lower_bound 50 \
    --preload_count 200 \
    --pin_mem \
    --peak_name "peaks_q0.01_tissue_open_exp" \
    --n_packs 1 \
    --flash_attn \
    --lr 5e-4 \
    --opt adamw \
    --wandb_project_name "get_finetune" \
    --wandb_run_name "ATACSplitPool_finetune_maxdepth" \
    --eval_freq 1 \
    --eval \
    --dist_eval \
    --eval_nonzero \
    --leave_out_celltypes "Astrocyte" \
    --leave_out_chromosomes "chr11" \
    --criterion "poisson" \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 5 \
    --epochs 100 \
    --num_region_per_sample 200 \
    --output_dir ${OUTPUT_DIR} 