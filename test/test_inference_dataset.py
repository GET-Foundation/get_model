#%%
import torch
from get_model.inference_engine import InferenceEngine
#%%
# Configuration for Gencode
gencode_config = {
    "assembly": "hg38",
    "version": 40,
    "gtf_dir": "/manitou/pmg/users/xf2217/bmms/caesar/data/"
}

# Configuration for the dataset
dataset_config = {
    "zarr_dirs": ["/pmglocal/xf2217/get_data/shendure_fetal_dense.zarr"],
    "genome_seq_zarr": "/pmglocal/xf2217/get_data/hg38.zarr",
    "genome_motif_zarr": "/pmglocal/xf2217/get_data/hg38_motif_result.zarr",
    "insulation_paths": [
        "/manitou/pmg/users/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather",
        "/manitou/pmg/users/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather"
    ],
    "peak_name": "peaks_q0.01_tissue_open_exp",
    "additional_peak_columns": ["Expression_positive", "Expression_negative", "aTPM", "TSS"],
    "n_peaks_upper_bound": 100
}

# Path to the model checkpoint
model_checkpoint = "/pmglocal/xf2217/20240221.conv50.atac_loss.nofreeze.nodepth.R100L1000/checkpoint-best.pth"
#%%
# Initialize the InferenceEngine
engine = InferenceEngine(gencode_config, dataset_config, model_checkpoint)
#%%
# Specify the gene and cell type for inference
gene_name = "MIA3"
celltype = 'Fetal Erythroblast 2.shendure_fetal.sample_40_liver.2048'  # Update this with your actual cell type

# Run inference
inference_results, prepared_batch = engine.run_inference_for_gene_and_celltype(gene_name, celltype)

# Handle or display the inference results as needed
print("Inference results:", inference_results)
# Additional result processing can go here
#%%
