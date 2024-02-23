#%%
import random

import numpy as np
import pandas as pd
import seaborn as sns
from caesar.io.gencode import Gencode
from caesar.io.zarr_io import DenseZarrIO
from pyranges import PyRanges as pr
from sympy import var
from tqdm import tqdm

from get_model.dataset.zarr_dataset import InferenceDataset
from get_model.inference_engine import InferenceEngine

random.seed(0)

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
#%%

def load_erythroblast_abe_data(csv_path='Editable_A_scores.combined.scores.csv'):
    base_edit = pd.read_csv('Editable_A_scores.combined.scores.csv')
    base_edit['Chromosome'] = base_edit['coord'].str.split(':').str[0]
    base_edit['Start'] = base_edit['coord'].str.split(':').str[1].str.split('-').str[0].astype(int)
    base_edit['End'] = base_edit['coord'].str.split(':').str[1].str.split('-').str[1].str[:-1].astype(int)
    base_edit['Strand'] = base_edit['coord'].str[-1]
    base_edit['Ref'] = 'A' if base_edit['Strand'].values[0] == '+' else 'T'
    base_edit['Alt'] = 'G' if base_edit['Strand'].values[0] == '+' else 'C'
    return base_edit, base_edit[['Chromosome', 'Start', 'End', 'Ref', 'Alt']]

base_edit_annot, base_edit = load_erythroblast_abe_data()
#%%
gencode = Gencode(**gencode_config)
dataset = InferenceDataset(**dataset_config, gencode_obj=gencode)
#%%
# Path to the model checkpoint
model_checkpoint = "/burg/home/xf2217/checkpoint-99.pth"
#%%
hg38 = DenseZarrIO('/pmglocal/xf2217/get_data/hg38.zarr')
# %%
# random generate variants in chr8:126000000-130000000
def one_hot_to_dna(one_hot_array):
    """Converts a one-hot encoded DNA sequence back to a string representation."""
    # Define the mapping from one-hot encoding to nucleotides
    mapping = {tuple([1, 0, 0, 0]): 'A', tuple([0, 1, 0, 0]): 'C', tuple([0, 0, 1, 0]): 'G', tuple([0, 0, 0, 1]): 'T', tuple([0, 0, 0, 0]): 'N'}
    # Convert each one-hot encoded nucleotide back to its string representation
    dna_sequence = ''.join(mapping[tuple(row)] for row in one_hot_array)
    return dna_sequence

def generate_variant_in(chr, start, end, n=1000):
    variants = []
    for i in tqdm(range(n)):
        s = random.randint(start, end)
        e = s + 1
        ref = one_hot_to_dna(hg38.get_track(chr, s, e))
        alt = random.choice(list(set(['A', 'T', 'C', 'G'])-set(ref)))
        variants.append((chr, s, e, ref, alt))
    variants = pd.DataFrame(variants, columns=['Chromosome', 'Start', 'End', 'Ref', 'Alt'])
    return variants


# %%
# Initialize the InferenceEngine
# Specify the gene and cell type for inference
gene_name = "BCL11A"
celltype = 'Fetal Erythroblast 1.shendure_fetal.sample_7_liver.4096'  # Update this with your actual cell type
engine = InferenceEngine(dataset, model_checkpoint)
engine.setup_data(gene_name, celltype)
strand = engine.gene_info.query('gene_name==@gene_name').Strand.map({'+':0, '-':1}).values[0]
# variants 

# engine_mut.setup_data(gene_name, celltype)

#%%
pred_exps = []
pred_exps_mut = []
obs_exps = []
# Run inference
for i, variants in tqdm(base_edit.iterrows()):
    engine_mut = InferenceEngine(dataset, model_checkpoint, mut=variants)
    engine_mut.setup_data(gene_name, celltype)
    inference_results, prepared_batch, tss_peak = engine.run_inference_for_gene_and_celltype(offset=0)
    inference_results_mut, prepared_batch_mut, tss_peak_mut = engine_mut.run_inference_for_gene_and_celltype(offset=0)

    pred_exp, ob_exp, pred_atac, ob_atac = inference_results
    pred_exp = pred_exp.reshape(-1, 2)
    ob_exp = ob_exp.reshape(-1, 2)
    pred_exp_mut, ob_exp_mut, pred_atac_mut, ob_atac_mut = inference_results_mut
    pred_exp_mut = pred_exp_mut.reshape(-1, 2)
    ob_exp_mut = ob_exp_mut.reshape(-1, 2)

    pred_exps.append(pred_exp[tss_peak, strand].sum())
    pred_exps_mut.append(pred_exp_mut[tss_peak_mut, strand].sum())
    obs_exps.append(ob_exp[tss_peak, strand].max())
# %%

sns.histplot(np.array([s.mean() for s in pred_exps]), bins=50, color='blue', alpha=0.5)
# sns.histplot(np.array([s.mean() for s in pred_exps_mut]), bins=50, color='red', alpha=0.5)
sns.histplot(np.array([s.mean() for s in obs_exps]), bins=50, color='black', alpha=0.5)
# %%
sns.scatterplot(x=pred_atac_mut, y=pred_atac)
# %%
prepared_batch_ = prepared_batch
# %%
engine.setup_data(gene_name, celltype)
inference_results, prepared_batch, tss_peak = engine.run_inference_for_gene_and_celltype(offset=0)
# %%
prepared_batch_[2]
# %%
prepared_batch[2]
# %%
sum(prepared_batch[0]==prepared_batch_[0])
# %%
(prepared_batch[1]==prepared_batch_[1])[0].sum(0)
# %%
(prepared_batch[3]==prepared_batch_[3])
# %%
prepared_batch[3]
# %%
