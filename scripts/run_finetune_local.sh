#!/bin/bash
# source /manitou-home/home/xf2217/.bashrc
# mamba activate /manitou/pmg/users/xf2217/mambaforge/atac_rna_data_processing
# mkdir -p /pmglocal/xf2217/pretrain_human_bingren_shendure_apr2023/
# cp -r /manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/k562_cut  /pmglocal/xf2217/pretrain_human_bingren_shendure_apr2023/
# cp -r /manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/k562_encode  /pmglocal/xf2217/pretrain_human_bingren_shendure_apr2023/
# cp -r /manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/k562_count_10  /pmglocal/xf2217/pretrain_human_bingren_shendure_apr2023/
# cp -r /manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/data  /pmglocal/xf2217/pretrain_human_bingren_shendure_apr2023/
# Set the path to save checkpoints
OUTPUT_DIR='/pmglocal/alb2281/repos/get_model/finetune_natac_test/'
# path to expression set
DATA_PATH='/pmglocal/xf2217/pretrain_human_bingren_shendure_apr2023/'
PORT=7956


# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=4 finetune.py \
    --data_set "Expression" \
    --data_path ${DATA_PATH} \
    --input_dim 283 \
    --eval_freq 1 \
    --criterion "poisson" \
    --data_type k562_cut \
    --model get_finetune_motif \
    --use_natac \
    --resume /pmglocal/xf2217/finetune_natac_test/pretrain_finetune_natac_fetal_adult.pth \
    --batch_size 64 \
    --leave_out_celltypes "k562_cut0.04" \
    --leave_out_chromosomes "chr11" \
    --lr 1e-3 \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --epochs 100 \
    --num_region_per_sample 200 \
    --output_dir ${OUTPUT_DIR} \
    --wandb_project_name "get_finetune" \
    --wandb_run_name "debug"
