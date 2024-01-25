#!/bin/bash
# Set the path to save checkpoints
OUTPUT_DIR='/pmglocal/alb2281/get-finetune/ckpts'
# path to expression set
DATA_PATH='/pmglocal/alb2281/htan_data/htan_zarr_final/'
PORT=7961

export NCCL_P2P_LEVEL=NVL

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=2 python -m torch.distributed.run --nproc_per_node=2 --rdzv-endpoint=localhost:$PORT /pmglocal/alb2281/repos/get_model/get_model/finetune.py \
    --data_set "HTAN_GBM" \
    --eval_data_set "HTAN_GBM.eval" \
    --finetune "/pmglocal/alb2281/get_ckpts/checkpoint-135.pth" \
    --data_path ${DATA_PATH} \
    --input_dim 639 \
    --output_dim 2 \
    --num_motif 637 \
    --model get_finetune_motif \
    --batch_size 32 \
    --num_workers 64 \
    --n_peaks_lower_bound 20 \
    --n_peaks_upper_bound 100 \
    --preload_count 200 \
    --pin_mem \
    --peak_name "peaks_q0.01_tissue_open_exp" \
    --n_packs 1 \
    --lr 5e-4 \
    --opt adamw \
    --wandb_project_name "get_finetune" \
    --wandb_run_name "debug" \
    --eval_freq 1 \
    --dist_eval \
    --eval_nonzero \
    --leave_out_celltypes "Oligodendrocytes" \
    --leave_out_chromosomes "chr11" \
    --criterion "poisson" \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 5 \
    --epochs 200 \
    --num_region_per_sample 100 \
    --output_dir ${OUTPUT_DIR} 
