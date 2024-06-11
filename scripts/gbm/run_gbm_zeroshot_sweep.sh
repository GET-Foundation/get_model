project_name="get-zeroshot-gbm-hparam-sweep"
zeroshot_sample="Tumor.htan_gbm.C3L-03405_CPT0224600013_snATAC_GBM_Tumor.2048"


# zero-shot with natac ckpt
for count_filter in 0 1 2 3 4 5 6 7 8 9 10; do 
    for motif_scaler in 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5; do
        for sample in $zeroshot_sample; do
            zeroshot_run_name="gbm_zeroshot_natac_countfilter_${count_filter}_motifscaler_${motif_scaler}_sample_${sample}"
            echo "Starting [$zeroshot_run_name]..."
            CUDA_VISIBLE_DEVICES=0,1 python /pmglocal/alb2281/repos/get_model/get_model/debug/debug_run_ref_region.py \
                +machine=manitou_alb2281 \
                stage=validate \
                dataset.reference_region_motif.count_filter=$count_filter \
                dataset.reference_region_motif.motif_scaler=$motif_scaler \
                dataset.quantitative_atac=False \
                dataset.keep_celltypes=$sample \
                finetune.checkpoint=/burg/pmg/users/xf2217/get_checkpoints/Astrocytes_natac/checkpoint-best.pth \
                finetune.pretrain_checkpoint=False \
                wandb.run_name=$zeroshot_run_name \
                wandb.project_name=$project_name \
                machine.num_devices=2
        done
    done
done
