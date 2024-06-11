project_name="get-zeroshot-gbm-finetune-leaveout-sample"
zeroshot_sample="Tumor.htan_gbm.C3L-03405_CPT0224600013_snATAC_GBM_Tumor"


# finetune on single sample
for count_filter in 2; do
    for motif_scaler in 1.3; do
        finetune_run_name="gbm_finetune_natac_countfilter_${count_filter}_motifscaler_${motif_scaler}_${finetune_sample}"
        echo "Starting [$finetune_run_name]..."
        CUDA_VISIBLE_DEVICES=0,1 python /pmglocal/alb2281/repos/get_model/get_model/debug/debug_run_ref_region.py \
            +machine=manitou_alb2281 \
            stage=fit \
            dataset.reference_region_motif.count_filter=$count_filter \
            dataset.reference_region_motif.motif_scaler=$motif_scaler \
            dataset.quantitative_atac=False \
            dataset.keep_celltypes=$finetune_sample \
            finetune.checkpoint=/burg/pmg/users/xf2217/get_checkpoints/Astrocytes_natac/checkpoint-best.pth \
            finetune.pretrain_checkpoint=False \
            finetune.strict=False \
            wandb.run_name=$finetune_run_name \
            wandb.project_name=$project_name \
            machine.num_devices=2
    done
done
