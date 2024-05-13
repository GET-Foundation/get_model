set -e


project_name="get-zeroshot-gbm-all-pts"
sample_list_file="/pmglocal/alb2281/scratch/zero_shot_samples_cont.txt"
mapfile -t samples < "$sample_list_file"

# # zero-shot with natac ckpt
for count_filter in 2; do 
    for motif_scaler in 1.3; do
        for sample in "${samples[@]}"; do
            zeroshot_run_name="gbm_zeroshot_natac_countfilter_${count_filter}_motifscaler_${motif_scaler}_sample_${sample}"
            echo "Starting [$zeroshot_run_name]..."
            CUDA_VISIBLE_DEVICES=1 python /pmglocal/alb2281/repos/get_model/get_model/debug/debug_run_ref_region.py \
                +machine=manitou_alb2281 \
                stage=validate \
                dataset.reference_region_motif.count_filter=$count_filter \
                dataset.reference_region_motif.motif_scaler=$motif_scaler \
                dataset.quantitative_atac=False \
                dataset.keep_celltypes=$sample \
                finetune.checkpoint=/burg/pmg/users/xf2217/get_checkpoints/Astrocytes_natac/checkpoint-best.pth \
                wandb.run_name=$zeroshot_run_name \
                wandb.project_name=$project_name \
                machine.num_devices=1
        done
    done
done


# for count_filter in 2; do
#     for motif_scaler in 1.3; do
#         for chr_num in {1..22}; do
#             chr_str="chr${chr_num}"
#             finetune_run_name="gbm_finetune_count_filter_${count_filter}_motif_scaler_${motif_scaler}_sample_${sample}_ckpt_natac_leaveout_${chr_str}"
#             echo "Starting [$finetune_run_name]..."
#             CUDA_VISIBLE_DEVICES=0,1,2,3 python /pmglocal/alb2281/repos/get_model/get_model/debug/debug_run_ref_region.py \
#                 +machine=manitou_alb2281 \
#                 stage=fit \
#                 dataset.reference_region_motif.count_filter=$count_filter \
#                 dataset.reference_region_motif.motif_scaler=$motif_scaler \
#                 dataset.quantitative_atac=False \
#                 dataset.leave_out_chromosomes=$chr_str \
#                 finetune.checkpoint=/burg/pmg/users/xf2217/get_checkpoints/Astrocytes_natac/checkpoint-best.pth \
#                 finetune.pretrain_checkpoint=False \
#                 finetune.strict=False \
#                 wandb.run_name=$finetune_run_name \
#                 wandb.project_name=$project_name \
#                 machine.num_devices=4
#         done
#     done
# done


# for count_filter in 2; do
#     for motif_scaler in 1.3; do
#         for chr_str in X; do
#             finetune_run_name="gbm_finetune_count_filter_${count_filter}_motif_scaler_${motif_scaler}_sample_${sample}_ckpt_natac_leaveout_${chr_str}"
#             echo "Starting [$finetune_run_name]..."
#             CUDA_VISIBLE_DEVICES=0,1,2,3 python /pmglocal/alb2281/repos/get_model/get_model/debug/debug_run_ref_region.py \
#                 +machine=manitou_alb2281 \
#                 stage=fit \
#                 dataset.reference_region_motif.count_filter=$count_filter \
#                 dataset.reference_region_motif.motif_scaler=$motif_scaler \
#                 dataset.quantitative_atac=False \
#                 dataset.leave_out_chromosomes=$chr_str \
#                 finetune.checkpoint=/burg/pmg/users/xf2217/get_checkpoints/Astrocytes_natac/checkpoint-best.pth \
#                 finetune.pretrain_checkpoint=False \
#                 finetune.strict=False \
#                 wandb.run_name=$finetune_run_name \
#                 wandb.project_name=$project_name \
#                 machine.num_devices=4
#         done
#     done
# done
