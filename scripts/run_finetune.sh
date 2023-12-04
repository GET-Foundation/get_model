#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --time=8:00:00
#SBATCH --partition=pmg
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G

source /manitou-home/home/xf2217/.bashrc
mamba activate /manitou/pmg/users/xf2217/mambaforge/atac_rna_data_processing
mkdir -p /pmglocal/xf2217/pretrain_human_bingren_shendure_apr2023/
cp -r /manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/k562_cut  /pmglocal/xf2217/pretrain_human_bingren_shendure_apr2023/
cp -r /manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/k562_encode  /pmglocal/xf2217/pretrain_human_bingren_shendure_apr2023/
cp -r /manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/k562_count_10  /pmglocal/xf2217/pretrain_human_bingren_shendure_apr2023/
cp -r /manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/data  /pmglocal/xf2217/pretrain_human_bingren_shendure_apr2023/
# Set the path to save checkpoints
OUTPUT_DIR='/pmglocal/xf2217/output_e1600_r200_finetune_k562_slurm_1000/'
# path to expression set
DATA_PATH='/pmglocal/xf2217/pretrain_human_bingren_shendure_apr2023/'
PORT=7956

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 torchrun --nproc_per_node=4  /manitou/pmg/users/xf2217/get_model/finetune.py \
    --data_set "Expression" \
    --mask_tss \
    --data_path ${DATA_PATH} \
    --input_dim 283 \
    --eval_freq 5 \
    --criterion "poisson" \
    --data_type k562_cut,k562_count_10,k562_encode \
    --model get_finetune_motif \
    --batch_size 4 \
    --leave_out_celltypes "k562_cut0.04" \
    --leave_out_chromosomes "chr11" \
    --lr 1e-3 \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --epochs 100 \
    --num_region_per_sample 200 \
    --output_dir ${OUTPUT_DIR} 
