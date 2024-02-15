#!/bin/bash
# Set the path to save checkpoints
OUTPUT_DIR='/pmglocal/xf2217/20240205.finetune_conv50_depth1024_500_region_200bp/'
# path to expression set
DATA_PATH='/pmglocal/xf2217/get_data/'
PORT=7957

export NCCL_P2P_LEVEL=NVL

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 --rdzv-endpoint=localhost:$PORT get_model/finetune.py \
    --data_set "Expression_Finetune_Fetal" \
    --eval_data_set "Expression_Finetune_Fetal.fetal_eval" \
    --finetune "/pmglocal/xf2217/get_data/checkpoint-175.pth" \
    --data_path ${DATA_PATH} \
    --input_dim 639 \
    --output_dim 2 \
    --num_motif 637 \
    --model get_finetune_motif_with_atac \
    --batch_size 16 \
    --num_workers 32 \
    --n_peaks_lower_bound 20 \
    --n_peaks_upper_bound 500 \
    --center_expand_target 200 \
    --non_redundant 'depth_1024' \
    --preload_count 200 \
    --pin_mem \
    --peak_name "peaks_q0.01_tissue_open_exp" \
    --n_packs 1 \
    --lr 5e-4 \
    --opt adamw \
    --wandb_project_name "get_finetune" \
    --wandb_run_name "EvalTSS.OODChr4&14.20240213.finetune_from_175_conv50.depth1024_500_region_200bp" \
    --eval_freq 2 \
    --freeze_atac_attention \
    --dist_eval \
    --eval_nonzero \
    --leave_out_celltypes "Astrocyte" \
    --leave_out_chromosomes "chr4,chr14" \
    --criterion "poisson" \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 20 \
    --epochs 100 \
    --num_region_per_sample 500 \
    --output_dir ${OUTPUT_DIR} 
