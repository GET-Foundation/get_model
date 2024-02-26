#!/bin/bash
# Set the path to save checkpoints
DATE=`date +%Y%m%d`
OUTPUT_DIR="/pmglocal/xf2217/Expression_Finetune_K562.Chr4&14.conv50.atac_loss.nofreeze.use_insulation.nodepth.gap50.shift10.R100L1000.augmented.${DATE}/"
# path to expression set
DATA_PATH='/pmglocal/xf2217/get_data/'
PORT=7957

export NCCL_P2P_LEVEL=NVL
#"/burg/pmg/users/xf2217/get_data/checkpoint-399.from_shentong.R200L500.pth" \

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 torchrun --nproc_per_node=6 --rdzv-endpoint=localhost:$PORT get_model/finetune.py \
    --data_set "Expression_Finetune_K562.Chr4&14" \
    --eval_data_set "Expression_Finetune_K562.Chr4&14.Eval" \
    --finetune "/burg/pmg/users/xf2217/get_checkpoints/EvalTSS.AllChr.fetal_hsc_gbm.conv50.atac_loss.nofreeze.use_insulation.nodepth.gap50.shift10.R100L1000.from_sequence.20240224.344.pth" \
    --data_path ${DATA_PATH} \
    --input_dim 639 \
    --output_dim 2 \
    --num_motif 637 \
    --model get_finetune_motif_with_atac \
    --batch_size 32 \
    --num_workers 32 \
    --use_insulation \
    --n_peaks_lower_bound 10 \
    --n_peaks_upper_bound 100 \
    --center_expand_target 1000 \
    --preload_count 200 \
    --peak_inactivation 'random_tss' \
    --random_shift_peak \
    --invert_peak 0.1 \
    --pin_mem \
    --peak_name "peaks_q0.01_tissue_open_exp" \
    --save_ckpt_freq 5 \
    --n_packs 1 \
    --lr 1e-3 \
    --opt adamw \
    --wandb_project_name "get_finetune.st_checkpoint399" \
    --wandb_run_name "Expression_Finetune_K562.Chr4&14.conv50.atac_loss.nofreeze.use_insulation.nodepth.gap50.shift10.R100L1000.augmented."${DATE} \
    --eval_freq 2 \
    --dist_eval \
    --eval_tss \
    --leave_out_chromosomes "chr4,chr14" \
    --criterion "poisson" \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 20 \
    --epochs 100 \
    --num_region_per_sample 100 \
    --output_dir ${OUTPUT_DIR}
