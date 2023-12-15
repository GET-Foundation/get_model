#!/bin/bash
OUTPUT_DIR='/pmglocal/alb2281/repos/get_model/output/'
DATA_PATH='/pmglocal/alb2281/repos/get_model/data/pretrain_B_ALL_2023/'
PORT=7956

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=2 /pmglocal/alb2281/repos/get_model/get_model/finetune.py \
    --data_set "Expression" \
    --data_path ${DATA_PATH} \
    --input_dim 283 \
    --eval_freq 1 \
    --criterion "poisson" \
    --data_type B_ALL \
    --model get_finetune_motif \
    --use_natac \
    --finetune /pmglocal/alb2281/repos/get_model/ckpts/checkpoint-1579.pth \
    --batch_size 64 \
    --leave_out_celltypes "MH3266" \
    --leave_out_chromosomes "chr11" \
    --num_workers 0\
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --epochs 100 \
    --lr 1e-3 \
    --num_region_per_sample 200 \
    --output_dir ${OUTPUT_DIR} \
    --wandb_project_name "get_finetune_all" \
    --wandb_run_name "exp_0" > /pmglocal/alb2281/repos/get_model/logs/finetune_output.txt
