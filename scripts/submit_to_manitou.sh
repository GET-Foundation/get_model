#!/bin/bash
#SBATCH --job-name=get_finetune
#SBATCH --output=/pmglocal/xf2217/%x_%j.out                # Output file
#SBATCH --error=/pmglocal/xf2217/%x_%j.err                 # Error file
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --account pmg
#SBATCH --gpus-per-node=6
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=4G

# Set the root and data directories
ROOT_DIR=/pmglocal/xf2217/GET_STARTED
DATA_DIR=/burg/pmg/users/xf2217/get_data

module load mamba

# Create the root directory if it does not exist
if [ ! -d $ROOT_DIR ]; then
  mkdir -p $ROOT_DIR
fi

# Navigate to the root directory
cd $ROOT_DIR

CODEBASE_DIR=$ROOT_DIR/get_model
# Define and clone the get_model repository
if [ ! -d $CODEBASE_DIR ]; then
  git clone git@github.com:fuxialexander/get_model.git $CODEBASE_DIR
fi
cd $CODEBASE_DIR
git checkout finetune-with-atac
git pull --rebase

# Create and activate a new mamba environment
if [ ! -d ${ROOT_DIR}/mambaforge/get_started ]; then
  mamba env create -f ${CODEBASE_DIR}/environment.yml -p ${ROOT_DIR}/mambaforge/get_started
source activate ${ROOT_DIR}/mambaforge/get_started

# Copy and decompress data
if [ ! -d ${ROOT_DIR}/get_data ]; then
  cp -r $DATA_DIR ${ROOT_DIR}/get_data
  cd ${ROOT_DIR}/get_data
  for f in *.tar; do tar -xvf $f; done
fi

# Return to the codebase directory and install the package
cd $CODEBASE_DIR
pip install -e .

# Clone and install the caesar repository
cd $ROOT_DIR
git clone git@github.com:fuxialexander/caesar.git
cd caesar
pip install -e .

# Clone and install the atac_rna_data_processing repository
cd $ROOT_DIR
git clone git@github.com:fuxialexander/atac_rna_data_processing.git
cd atac_rna_data_processing
pip install -e .

# Return to the codebase directory
cd $CODEBASE_DIR



# Set the path to save checkpoints
DATE=`date +%Y%m%d`
OUTPUT_DIR="/burg/pmg/users/xf2217/get_checkpoints/Expression_Finetune_K562_HSC.Chr4&14.conv50.atac_loss.nofreeze.nodepth.gap50.shift10.R100L500.augmented.${DATE}/"
# path to expression set
DATA_PATH=${ROOT_DIR}/get_data
PORT=7957

export NCCL_P2P_LEVEL=NVL
#"/burg/pmg/users/xf2217/get_data/checkpoint-399.from_shentong.R200L500.pth" \

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 torchrun --nproc_per_node=6 --rdzv-endpoint=localhost:$PORT get_model/finetune.py \
    --data_set "Expression_Finetune_K562_HSC.Chr4&14" \
    --eval_data_set "Expression_Finetune_K562_HSC.Chr4&14.Eval" \
    --data_path ${DATA_PATH} \
    --input_dim 639 \
    --output_dim 2 \
    --num_motif 637 \
    --model get_finetune_motif_with_atac_hic \
    --batch_size 32 \
    --num_workers 32 \
    --n_peaks_lower_bound 10 \
    --n_peaks_upper_bound 100 \
    --center_expand_target 500 \
    --preload_count 200 \
    --peak_inactivation 'random_tss' \
    --random_shift_peak \
    --hic_path "${DATA_PATH}/4DNFI2TK7L2F.hic" \
    --pin_mem \
    --peak_name "peaks_q0.01_tissue_open_exp" \
    --save_ckpt_freq 5 \
    --n_packs 1 \
    --lr 1e-3 \
    --opt adamw \
    --wandb_project_name "Expression_Finetune_K562_HSC.hic" \
    --wandb_run_name "Expression_Finetune_K562_HSC.Chr4&14.conv50.atac_loss.nofreeze.nodepth.gap50.shift10.R100L1000.augmented."${DATE} \
    --eval_freq 2 \
    --dist_eval \
    --eval_tss \
    --leave_out_chromosomes "chr4,chr14" \
    --leave_out_celltypes "MPP|HSC|MKP|MEP|GMP" \
    --criterion "poisson" \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 20 \
    --epochs 100 \
    --num_region_per_sample 100 \
    --output_dir ${OUTPUT_DIR}
