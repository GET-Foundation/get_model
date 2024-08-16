#%%
from atac_rna_data_processing.io.celltype import GETHydraCellType
from atac_rna_data_processing.io.mutation import GETHydraCellMutCollection

from get_model.config.config import load_config, pretty_print_config
from get_model.run_region import run

#%%
# load config
cfg = load_config('fetal_region_erythroblast')
pretty_print_config(cfg)

#%%
# run model training
cfg.training.epochs=5 # use as demo there
run(cfg)
# %%
# Run inference to get the jacobian matrix for genes
# Setup config first. Change state to 'predict'
cfg.stage = 'predict'
cfg.machine.batch_size=2
cfg.wandb.run_name='fetal_region_erythroblast_inference'
cfg.task.gene_list='SOX10,MYC'
#%%
# run model inference
run(cfg)
# %%
# Load the inference result as a celltype object
hydra_celltype = GETHydraCellType.from_config(cfg)
hydra_celltype

# %%
# Get the jacobian matrix for MYC, summarize by region
hydra_celltype.get_gene_jacobian_summary('MYC', 'region').sort_values()
# %%
# Get the jacobian matrix for MYC, summarize by motif
hydra_celltype.get_gene_jacobian_summary('MYC', 'motif').sort_values()
# %%
# Run Mutation analysis for one mutation. Multiple mutations can be passed as a comma separated string like 'rs55705857,rs55705857'
cfg.machine.fasta_path = '/home/xf2217/Projects/common/hg38.fa'
cfg.task.mutations = 'rs55705857'
cell_mut_col = GETHydraCellMutCollection(cfg)
scores = cell_mut_col.get_all_variant_scores()
scores
# %%
