#!/bin/bash
project_name="get-gbm-dea-lora_natac"
checkpoint="/pmglocal/alb2281/get/get_ckpts/gbm_Tumor_C3L-03405_lora_natac.ckpt"
output_dir="/pmglocal/alb2281/get/output/natac-oneshot-gbm"

sample_list_file="/pmglocal/alb2281/get/get_data/gbm_samples_natac.txt"
# read the sample list file into an array
mapfile -t samples < "$sample_list_file"


for sample in "${samples[@]}"; do
    run_name="gbm_oneshot_gene_dea_${sample}_natac"
    echo "Starting [$run_name]..."
    CUDA_VISIBLE_DEVICES=0 python get_model/debug/debug_run_ref_region_gbm.py \
        +machine=manitou_alb2281 \
        +finetune=lightning \
        stage=predict \
        dataset.peak_count_filter=10 \
        dataset.reference_region_motif.motif_scaler=1.3 \
        machine.num_devices=1 \
        machine.batch_size=64 \
        machine.num_workers=8 \
        machine.output_dir=$output_dir \
        wandb.project_name=$project_name \
        wandb.run_name=$run_name \
        finetune.checkpoint=$checkpoint \
        finetune.strict=true \
        finetune.use_lora=true \
        dataset.keep_celltypes=$sample \
        task.gene_list=null
done
