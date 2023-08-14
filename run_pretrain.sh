#!/bin/bash
#SBATCH --job-name=pretrain_k562
#SBATCH --time=4:00:00
#SBATCH --partition=pmg
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G

source /manitou-home/home/xf2217/.bashrc
mamba activate /manitou/pmg/users/xf2217/mambaforge/atac_rna_data_processing
cp -r /manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/k562_cut  /pmglocal/xf2217/pretrain_human_bingren_shendure_apr2023/
cp -r /manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/data  /pmglocal/xf2217/pretrain_human_bingren_shendure_apr2023/
# Set the path to save checkpoints
OUTPUT_DIR='/pmglocal/xf2217/output_e1600_r200_pretrain_k562_cut_slurm/'
# path to expression set
DATA_PATH='/pmglocal/xf2217/pretrain_human_bingren_shendure_apr2023/'
PORT=7956

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2  pretrain.py \
    --data_set "Pretrain" \
    --data_path ${DATA_PATH} \
    --input_dim 283 \
    --data_type k562_cut \
    --mask_ratio 0.5 \
    --model get_pretrain_motif \
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

