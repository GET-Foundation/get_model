#!/bin/bash
set -e

# run one-shot gene DEA
project_name="get-gbm-oneshot-gene-dea"
checkpoint="/home/ubuntu/alb2281/get/get_ckpts/get_gbm_finetuned_C3L-03405_CPT0224600013.ckpt"

sample_list_file="/home/ubuntu/alb2281/get/get_data/gbm_zero_shot_samples.txt"
mapfile -t samples < "$sample_list_file"


for sample in "${samples[@]}"; do
    run_name="gbm_zeroshot_gene_dea_${sample}"
    echo "Starting [$run_name]..."
    CUDA_VISIBLE_DEVICES=0 python get_model/debug/debug_run_ref_region_gbm.py \
        +machine=manitou_alb2281 \
        stage=predict \
        dataset.peak_count_filter=10 \
        dataset.reference_region_motif.motif_scaler=1.3 \
        machine.num_devices=1 \
        machine.batch_size=64 \
        machine.num_workers=8 \
        wandb.project_name=$project_name \
        wandb.run_name=$run_name \
        finetune.checkpoint=$checkpoint \
        dataset.keep_celltypes=$sample \
        task.gene_list=null
done
