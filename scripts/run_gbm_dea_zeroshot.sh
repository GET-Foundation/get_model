#!/bin/bash
set -e

# run zero-shot gene DEA
project_name="get-gbm-zeroshot-gene-dea"
checkpoint="/home/ubuntu/alb2281/get/get_ckpts/astrocyte-natac-best.pth"

sample_list_file="/home/ubuntu/alb2281/get/get_data/gbm_zero_shot_samples.txt"
# read the sample list file into an array
mapfile -t samples < "$sample_list_file"


for sample in "${samples[@]}"; do
    # skip the first element in samples (already run)
    if [ "$sample" == "Tumor.htan_gbm.C3N-01818_CPT0168270014_snATAC_GBM_Tumor.2048" ]; then
        echo "Skipping [$sample]..."
        continue
    fi

    run_name="DEBUG_gbm_zeroshot_gene_dea_${sample}"
    echo "Starting [$run_name]..."
    CUDA_VISIBLE_DEVICES=0 python get_model/debug/debug_run_ref_region_gbm.py \
        +machine=manitou_alb2281 \
        stage=predict \
        dataset.peak_count_filter=10 \
        dataset.reference_region_motif.motif_scaler=1.3 \
        machine.num_devices=1 \
        machine.batch_size=16 \
        machine.num_workers=0 \
        wandb.project_name=$project_name \
        wandb.run_name=$run_name \
        finetune.checkpoint=$checkpoint \
        dataset.keep_celltypes=$sample \
        task.gene_list=null
done
