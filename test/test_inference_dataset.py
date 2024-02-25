#%%
import random

import numpy as np
import pandas as pd
import seaborn as sns
from caesar.io.gencode import Gencode
from caesar.io.zarr_io import DenseZarrIO
from pyranges import PyRanges as pr
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

#%%
def load_k562_crispr_data(csv_path='/burg/pmg/users/xf2217/CRISPR_comparison/resources/example/EPCrisprBenchmark_Fulco2019_K562_GRCh38.tsv.gz'):
    crispr = pd.read_csv(csv_path, sep='\t')
    crispr.rename(columns={'chrom':'Chromosome',
                           'chromStart': 'Start',
                           'chromEnd': 'End'}, inplace=True)
    return crispr
k562_crispr = load_k562_crispr_data()
#%%
k562_crispr
#%%
base_edit_annot, base_edit = load_erythroblast_abe_data()
#%%
gencode = Gencode(**gencode_config)
dataset = InferenceDataset(**dataset_config, gencode_obj=gencode)
#%%
# Path to the model checkpoint
model_checkpoint = "/burg/pmg/users/xf2217/get_checkpoints/fetal_hsc_gbm.all_chr.best.pth"
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
engine = InferenceEngine(dataset, model_checkpoint, with_sequence=False)
engine.setup_data('MYC', 'Fetal Erythroblast 2.shendure_fetal.sample_7_liver.4096')
#%%
chr_name, start, end, celltype_id, track, item_insulation, celltype_peaks, motif_mean_std  = engine.data
batch = engine.dataset.datapool.generate_sample(chr_name, start, end, engine.data_key, celltype_id, track, celltype_peaks, motif_mean_std=motif_mean_std, mut=None, peak_inactivation=None)
#%%
# Initialize the InferenceEngine
# Specify the gene and cell type for inference
def run_inference_for_gene_and_celltype(gene_name, celltype, base_edit_annot, dataset, model_checkpoint, with_sequence=False):
    engine = InferenceEngine(dataset, model_checkpoint, with_sequence=with_sequence)
    engine.setup_data(gene_name, celltype, window_idx_offset=0)
    strand = engine.gene_info.query('gene_name==@gene_name').Strand.map({'+':0, '-':1}).values[0]
    # variants 
    chromosome = engine.gene_info.query('gene_name==@gene_name').Chromosome.values[0]
    # engine_mut.setup_data(gene_name, celltype)
    base_edit_to_test = base_edit_annot.query('Chromosome==@chromosome')
    engine_mut = InferenceEngine(dataset, model_checkpoint, mut=base_edit_to_test, with_sequence=with_sequence)
    engine_mut.setup_data(gene_name, celltype, window_idx_offset=0)
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
        base_edit_to_test.loc[i, 'atac_fc'] = (pred_atac_mut-pred_atac).sum() 
    base_edit_to_test['Gene'] = gene_name
    return base_edit_to_test
#%%
celltype = 'Fetal Erythroblast 2.shendure_fetal.sample_7_liver.4096'  # Update this with your actual cell type

for gene in ['MYB', 'HBG1', 'HBG2', 'NFIX', 'KLF1' , 'BCL11A']:
    base_edit_to_test = run_inference_for_gene_and_celltype(gene, celltype, base_edit_annot, dataset, model_checkpoint)
    base_edit_to_test.to_csv(f'{gene}.HbFBase_inference_results.with_sequence.csv')

# %%
# Load all inference results
celltype = 'Fetal Erythroblast 2.shendure_fetal.sample_7_liver.4096'  # Update this with your actual cell type

inference_results = {}
for gene in  ['MYB', 'HBG1', 'HBG2', 'NFIX', 'KLF1' , 'BCL11A']:
    df = pd.read_csv(f'{gene}.HbFBase_inference_results.with_sequence.csv')
    df['exp_fc_abs'] = df['exp_fc'].abs()
    df['GET'] = df['exp_fc'] * df['atac_fc']
    inference_results[gene] = df
inference_results = pd.concat(inference_results.values()) 
peaks = dataset.datapool.zarr_dict['shendure_fetal_dense.zarr'].get_peaks(celltype, 'peaks_q0.01_tissue_open_exp')

#%%
data = pr(inference_results.dropna()).join(pr(peaks)).df
import seaborn as sns

sns.scatterplot(data=data,
                x='HbFBase', y='exp_fc', hue='aTPM')
# %%
#barplot of spearman correlation for DeepSEA, CADD, and GERP and exp_fc with HbFBase
correlations = {}
for col in ['DeepSEA', 'CADD', 'GERP', 'GET']:
    correlations[col] = data.eval(f'HbFBase.corr({col}, method="pearson")')

sns.barplot(x=list(correlations.keys()), y=list(correlations.values()))



#%%
threshold = 30
data = data#.query('HbFBase>30 or HbFBase<10')
def plot_aupr(data, gene, score, ax):
    from sklearn.metrics import auc, precision_recall_curve, roc_curve
    precision, recall, _ = precision_recall_curve(data['HbFBase']>threshold, data[score])
    # auroc
    fp_rate, tp_rate, _ = roc_curve(data['HbFBase']>threshold, data[score])
    ax.plot(recall, precision, label=f'{score} AUPR: {auc(recall, precision):.2f}')
    # ax.plot(fp_rate, tp_rate, linestyle='--', label=f'{score} AUROC: {auc(fp_rate, tp_rate):.2f}')
    # add random line 
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend()
    return ax

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
for score in ['DeepSEA', 'CADD', 'GERP', 'atac_fc']:
    plot_aupr(data, gene, score, ax)    
random_aupr = (data['HbFBase']>threshold).sum()/len(data)
ax.plot([0, 1], [(data['HbFBase']>threshold).sum()/len(data)]*2, linestyle='--', color = 'black', label=f'Random AUPR: {random_aupr:.2f}')
ax.legend()
plt.tight_layout()
# %%
