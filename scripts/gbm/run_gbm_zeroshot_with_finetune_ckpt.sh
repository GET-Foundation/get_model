project_name="get-zeroshot-gbm-finetuned-ckpt"
mapfile -t samples < "/pmglocal/alb2281/repos/get_model/scripts/gbm/gbm_max_depth_samples.txt" # list of max_depth samples


# zero-shot with natac ckpt
for count_filter in 2; do
    for motif_scaler in 1.3; do
        for sample in "${samples[@]}"; do
            zeroshot_run_name="gbm_zeroshot_finetuned_natac_countfilter_${count_filter}_motifscaler_${motif_scaler}_sample_${sample}"
            echo "Starting [$zeroshot_run_name]..."
            CUDA_VISIBLE_DEVICES=2,3 python /pmglocal/alb2281/repos/get_model/get_model/debug/debug_run_ref_region.py \
                +machine=manitou_alb2281 \
                stage=validate \
                dataset.reference_region_motif.count_filter=$count_filter \
                dataset.reference_region_motif.motif_scaler=$motif_scaler \
                dataset.quantitative_atac=False \
                dataset.keep_celltypes=$sample \
                finetune.checkpoint=/pmglocal/alb2281/get/get_ckpts/get_gbm_finetuned_C3L-03405_CPT0224600013.ckpt \
                finetune.pretrain_checkpoint=False \
                wandb.run_name=$zeroshot_run_name \
                wandb.project_name=$project_name \
                machine.num_devices=2
        done
    done
done
