project_name="get-oneshot-gene-dea"
sample_list_file="/manitou/pmg/users/alb2281/data/zero_shot_samples.txt"
mapfile -t samples < "$sample_list_file"

# # zero-shot with natac ckpt
for count_filter in 2; do 
    for motif_scaler in 1.3; do
        for sample in "Tumor.htan_gbm.C3N-01814_CPT0167860015_snATAC_GBM_Tumor.16384"; do
            zeroshot_run_name="gbm_zeroshot_natac_countfilter_${count_filter}_motifscaler_${motif_scaler}_sample_${sample}"
            echo "Starting [$zeroshot_run_name]..."
            CUDA_VISIBLE_DEVICES=0 python /pmglocal/alb2281/repos/get_model/get_model/debug/debug_run_ref_region.py \
                +machine=manitou_alb2281 \
                stage=predict \
                dataset.reference_region_motif.count_filter=$count_filter \
                dataset.reference_region_motif.motif_scaler=$motif_scaler \
                dataset.quantitative_atac=False \
                dataset.keep_celltypes=$sample \
                finetune.checkpoint=/pmglocal/alb2281/get/get_ckpts/finetuned/get_gbm_finetuned_C3L-03405_CPT0224600013.ckpt \
                wandb.run_name=$zeroshot_run_name \
                wandb.project_name=$project_name \
                machine.num_devices=1
        done
    done
done
