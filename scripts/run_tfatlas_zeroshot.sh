for peak_count_filter in {1..10}; do
    for motif_scaler in 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5; do
        find /home/xf2217/Projects/get_data/joung_tfatlas_dense.zarr -type d -name '*L10M*' | cut -d'/' -f7 | while read -r sample; do
            python get_model/debug/debug_run_ref_region.py \
                +machine=pc \
                stage=validate \
                dataset.peak_count_filter=$peak_count_filter \
                dataset.reference_region_motif.motif_scaler=$motif_scaler \
                finetune.checkpoint=/home/xf2217/Projects/get_checkpoints/Astrocytes_natac/checkpoint-best.pth \
                wandb.project_name="RunRefRegionTFAtlasZeroshotSweep" \
                wandb.run_name="$(basename "$sample")" \
                dataset.leave_out_celltypes="$(basename "$sample")"
        done
    done 
done