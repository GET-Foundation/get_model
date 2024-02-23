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
import pyliftover
def load_erythroblast_abe_data(csv_path='/burg/pmg/users/xf2217/Editable_A_scores.combined.scores.csv'):
    base_edit = pd.read_csv(csv_path)
    base_edit['Chromosome'] = base_edit['coord'].str.split(':').str[0]
    base_edit['Start'] = base_edit['coord'].str.split(':').str[1].str.split('-').str[0].astype(int)
    base_edit['End'] = base_edit['coord'].str.split(':').str[1].str.split('-').str[1].str[:-1].astype(int)
    base_edit['Strand'] = base_edit['coord'].str[-1]

    # liftover hg19 to hg38
    lo = pyliftover.LiftOver('hg19', 'hg38')
    base_edit['Start'] = [lo.convert_coordinate(ch, s)[0][1] for ch, s in base_edit[['Chromosome', 'Start']].values]
    base_edit['End'] = base_edit['Start']+1
    base_edit['Ref'] = base_edit['Strand'].map({'+':'A', '-':'T'})
    base_edit['Alt'] = base_edit['Strand'].map({'+':'G', '-':'C'})
    return base_edit, base_edit[['Chromosome', 'Start', 'End', 'Ref', 'Alt']]

base_edit_annot, base_edit = load_erythroblast_abe_data()
#%%
gencode = Gencode(**gencode_config)
dataset = InferenceDataset(**dataset_config, gencode_obj=gencode)
#%%
# Path to the model checkpoint
model_checkpoint = "/burg/home/xf2217/checkpoint-best.pth"
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
gene_name = "HBG1"
celltype = 'Fetal Erythroblast 2.shendure_fetal.sample_7_liver.4096'  # Update this with your actual cell type
engine = InferenceEngine(dataset, model_checkpoint)
engine.setup_data(gene_name, celltype, window_idx_offset=0)
strand = engine.gene_info.query('gene_name==@gene_name').Strand.map({'+':0, '-':1}).values[0]
# variants 
chromosome = engine.gene_info.query('gene_name==@gene_name').Chromosome.values[0]
# engine_mut.setup_data(gene_name, celltype)
base_edit_to_test = base_edit_annot.query('Chromosome==@chromosome')

engine_mut = InferenceEngine(dataset, model_checkpoint, mut=base_edit_to_test)
engine_mut.setup_data(gene_name, celltype, window_idx_offset=0)
#%%

pr(engine.peak_info).join(pr(base_edit_to_test))
#%%
# Run inference

for i, variants in tqdm(base_edit_to_test.iterrows()):
    inference_results, prepared_batch, tss_peak = engine.run_inference_for_gene_and_celltype(offset=0)
    inference_results_mut, prepared_batch_mut, tss_peak_mut = engine_mut.run_inference_for_gene_and_celltype(offset=0, mut=pd.DataFrame(variants).T)
    pred_exp, ob_exp, pred_atac, ob_atac = inference_results
    pred_exp = pred_exp.reshape(-1, 2)
    ob_exp = ob_exp.reshape(-1, 2)
    pred_exp_mut, ob_exp_mut, pred_atac_mut, ob_atac_mut = inference_results_mut
    pred_exp_mut = pred_exp_mut.reshape(-1, 2)
    ob_exp_mut = ob_exp_mut.reshape(-1, 2)
    ob_atac = ob_atac.reshape(-1)
    pred_atac = pred_atac.reshape(-1)
    pred_atac_mut = pred_atac_mut.reshape(-1)

    base_edit_to_test.loc[i, 'pred_exp'] = pred_exp[tss_peak, strand].mean()
    base_edit_to_test.loc[i, 'pred_exp_mut'] = pred_exp_mut[tss_peak_mut, strand].mean()
    base_edit_to_test.loc[i, 'obs_exp'] = ob_exp[tss_peak, strand].max()
    base_edit_to_test.loc[i, 'ob_atac'] = ob_atac[tss_peak_mut].max()
    base_edit_to_test.loc[i, 'pred_atac_mut'] = pred_atac_mut[tss_peak_mut].max()
    base_edit_to_test.loc[i, 'pred_atac'] = pred_atac[tss_peak].max()
    base_edit_to_test.loc[i, 'exp_fc'] = pred_exp_mut[tss_peak, strand].sum() - pred_exp[tss_peak, strand].sum()
    base_edit_to_test.loc[i, 'atac_fc'] = pred_atac_mut[tss_peak_mut].sum() - pred_atac[tss_peak].sum()
# %%
import seaborn as sns
sns.scatterplot(data=pr(base_edit_to_test.dropna()).join(pr(engine.peak_info)).df,
                x='HbFBase', y='atac_fc', hue='aTPM')
# %%
data = pr(base_edit_to_test.dropna()).join(pr(engine.peak_info)).df.query('aTPM>0.1')
data['group'] = ['HbFBase>30' if x>10 else 'HbFBase<30' for x in data['HbFBase']]
# violin plot of group HbFBase>30 and HbFBase<30
sns.violinplot(data=data, x='group', y='GERP', cut=0, inner='quartile')

# add y=0 line
#%%
# plot aupr for exp_fc, DeepSEA, CADD, and GERP against 'group'
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
def plot_aupr(data, ax, title, scores=['DeepSEA', 'CADD', 'GERP'], labels='group', hue=['exp_fc', ]):
    # calculate the average precision score for DeepSEA, CADD, and GERP, exp_fc
    for group, df in data.groupby(hue):
        precision, recall, _ = precision_recall_curve(df[x]>30, df[y])
        ax.plot(recall, precision, label=group)
    ax.set_title(title)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend()
    return ax

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plot_aupr(data, ax, 'exp_fc vs HbFBase', 'HbFBase', 'exp_fc')






#%%











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
