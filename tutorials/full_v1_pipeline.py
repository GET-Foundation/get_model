# Install necessary packages in the most simplistic setting. Ignore this if you follow the main installation instructions.
# ! pip install pandas zarr scipy numcodecs
# ! apt-get update && apt-get install -y bedtools
# NOTE: tabix has to be >= 1.17
# ! apt-get install gcc
# ! apt-get install make
# ! apt-get install libbz2-dev
# ! apt-get install zlib1g-dev
# ! apt-get install libncurses5-dev
# ! apt-get install libncursesw5-dev
# ! apt-get install liblzma-dev
# ! cd /usr/bin
# ! wget https://github.com/samtools/htslib/releases/download/1.17/htslib-1.17.tar.bz2
# ! tar -vxjf htslib-1.17.tar.bz2
# ! cd htslib-1.17 && ./configure && make && make install
# ! export PATH="$PATH:/usr/bin/htslib-1.17"
# ! source ~/.profile
#%%
import os

from atac_rna_data_processing.io.celltype import GETHydraCellType
from atac_rna_data_processing.io.mutation import GETHydraCellMutCollection
from preprocess_utils import (Gencode, add_atpm, add_exp, create_peak_motif,
                              download_motif, get_motif, join_peaks,
                              query_motif, unzip_zarr, zip_zarr)

from get_model.config.config import load_config, pretty_print_config
from get_model.dataset.zarr_dataset import (InferenceRegionMotifDataset,
                                            RegionMotifDataset)
from get_model.run_region import run_zarr as run
from get_model.utils import print_shape

# %%
# # Preprocess the data
# %%
motif_bed_url = "https://resources.altius.org/~jvierstra/projects/motif-clustering/releases/v1.0/hg38.archetype_motifs.v1.0.bed.gz"
motif_bed_index_url = "https://resources.altius.org/~jvierstra/projects/motif-clustering/releases/v1.0/hg38.archetype_motifs.v1.0.bed.gz.tbi"
peak_bed = "/home/xf2217/Projects/pretrain_human_bingren_shendure_apr2023/fetal_adult/118.atac.bed"
reference_peaks = None

if (
    motif_bed_url
    and motif_bed_index_url
    and not (
        os.path.exists("hg38.archetype_motifs.v1.0.bed.gz")
        or os.path.exists("hg38.archetype_motifs.v1.0.bed.gz.tbi")
    )
):
    download_motif(motif_bed_url, motif_bed_index_url)
    motif_bed = "hg38.archetype_motifs.v1.0.bed.gz"
else:
    motif_bed = "/home/xf2217/Projects/get_data/hg38.archetype_motifs.v1.0.bed.gz"
# %%
# join peaks with reference peaks if provided. Ideally, after doing this step, you should get
# the corresponding aTPM values again for the joint peaks using the scATAC data. For simplicity,
# we don't use the reference peaks here but it's optimal to have the pretraining peaks as reference when finetuning.
joint_peaks = join_peaks(peak_bed, reference_peaks)
# %%
# query motif and get motifs in the peaks
peaks_motif = query_motif(joint_peaks, motif_bed)
get_motif_output = get_motif(joint_peaks, peaks_motif)

# %%
# create peak motif zarr file
create_peak_motif(get_motif_output, "output.zarr")
# %%
# add aTPM data for multiple cell types
add_atpm(
    "output.zarr",
    "astrocyte.atac.bed",
    "astrocyte",
)
# %%
# add expression and TSS data for multiple cell types
add_exp(
    "output.zarr",
    "astrocyte.rna.csv",
    "astrocyte.atac.bed",
    "astrocyte",
)
# %%
# optionally zip the zarr file for download or storage
zip_zarr("output.zarr")
# %%
# clean up intermediate files
for file in [joint_peaks, peaks_motif, get_motif_output]:
    os.remove(file)
# %%
# Unzip if necessary
unzip_zarr("output.zarr")
#%%
# load the zarr file as a dataset. 
region_motif_dataset = RegionMotifDataset(
    "./output.zarr",
    celltypes="astrocyte",
    quantitative_atac=True,
    num_region_per_sample=200,
    leave_out_celltypes=None,
    leave_out_chromosomes="chr11",
    is_train=False,
)
# %%
print_shape(region_motif_dataset[0])
# %%
# load the zarr file as a inference dataset, where you only focus on the genes of interest. 
gencode = Gencode(assembly="hg38", version=40)
inference_region_motif_dataset = InferenceRegionMotifDataset(
    zarr_path="./output.zarr",
    gencode_obj={'hg38':gencode},
    assembly='hg38',
    gene_list='RET,MYC,BCL11A',
    celltypes="astrocyte",
    quantitative_atac=True,
    num_region_per_sample=200,
    leave_out_celltypes=None,
    leave_out_chromosomes=None,
    is_train=True,
)
#%%
print_shape(inference_region_motif_dataset[1])

#%%
# # Finetune the model
#%%
# load config
cfg = load_config('finetune_tutorial')
pretty_print_config(cfg)
cfg.run.run_name='training'
cfg.dataset.zarr_path = "./output.zarr"
#%%
# run model training
cfg.training.epochs=5 # use as demo there
trainer = run(cfg)
#%%
# Run inference to get the jacobian matrix for genes
# Setup config first. Change state to 'predict'
cfg.stage = 'predict'
cfg.machine.batch_size=1
# resume from the best checkpoint we just traineds
cfg.finetune.resume_ckpt = trainer.checkpoint_callback.best_model_path
cfg.run.run_name='interpret'
#%%
# run model inference
run(cfg)
# %%
# Load the inference result as a celltype object
hydra_celltype = GETHydraCellType.from_config(cfg)
hydra_celltype

# %%
# Get the jacobian matrix for MYC, summarize by region
hydra_celltype.get_gene_jacobian_summary('MYC', 'region')
# %%
# Get the jacobian matrix for MYC, summarize by motif
hydra_celltype.get_gene_jacobian_summary('MYC', 'motif').sort_values()
# %%
# Run Mutation analysis for one mutation. Multiple mutations can be passed as a comma separated string like 'rs55705857,rs55705857'
#%%
# download hg38 fasta file from ucsc
! wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
# unzip
! gunzip hg38.fa.gz
#%%
cfg.machine.fasta_path = 'hg38.fa'
cfg.task.mutations = 'rs55705857'
cfg.dataset.leave_out_celltypes = 'astrocyte'
cell_mut_col = GETHydraCellMutCollection(cfg)
# %%
cell_mut_col.variant_to_genes
# %%
scores = cell_mut_col.get_all_variant_scores()
# %%
scores
# %%