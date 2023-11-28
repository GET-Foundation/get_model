
import sys
sys.path.append('/manitou/pmg/users/xf2217/atac_rna_data_processing/')
import os
import re

import numpy as np
from atac_rna_data_processing.io.causal_lib import *

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from atac_rna_data_processing.config.load_config import load_config
from atac_rna_data_processing.io.mutation import read_rsid, read_rsid_parallel, Mutations, MutationsInCellType
from atac_rna_data_processing.io.nr_motif_v1 import NrMotifV1
from atac_rna_data_processing.io.celltype import GETCellType
from pyranges import PyRanges as pr
import  matplotlib.pyplot as plt

plt.style.use('manuscript.mplstyle')
#%%
# from adhoc_model_loading import *
#%%
GET_CONFIG = load_config('/manitou/pmg/users/xf2217/atac_rna_data_processing/atac_rna_data_processing/config/GET')
GET_CONFIG.celltype.jacob=False
GET_CONFIG.celltype.num_cls=2
GET_CONFIG.celltype.input=True
GET_CONFIG.celltype.embed=False
GET_CONFIG.assets_dir=''
GET_CONFIG.s3_file_sys=''
GET_CONFIG.celltype.data_dir = '../pretrain_human_bingren_shendure_apr2023/fetal_adult/'
GET_CONFIG.celltype.interpret_dir='/manitou/pmg/users/xf2217/Interpretation_all_hg38_allembed_v4_natac'
cell_type_annot = pd.read_csv(os.path.join(GET_CONFIG.celltype.data_dir, '../data/cell_type_pretrain_human_bingren_shendure_apr2023.txt'))#.set_index('id').celltype.to_dict()
cell_type_annot_dict = cell_type_annot.set_index('id').celltype.to_dict()
#%%
# AdultAst1 = GETCellType('108', GET_CONFIG)
# FetalAst1 = GETCellType('118', GET_CONFIG)
# FetalAst2 = GETCellType('129', GET_CONFIG)
# FetalAst3 = GETCellType('178', GET_CONFIG)
# FetalAst4 = GETCellType('217', GET_CONFIG)
# AdultOligP = GETCellType('4', GET_CONFIG)
# AdultOlig = GETCellType('38', GET_CONFIG)
# AdultOlig.get_gene_by_motif()
# AdultOligP.get_gene_by_motif()
# AdultAst1.get_gene_by_motif()
# FetalAst1.get_gene_by_motif()
# FetalAst2.get_gene_by_motif()
# FetalAst3.get_gene_by_motif()
# FetalAst4.get_gene_by_motif()
#%%
from atac_rna_data_processing.io.region import *

hg38 = Genome('hg38', 'hg38.fa')

#%%
motif = NrMotifV1.load_from_pickle('./NrMotifV1.pkl')
#%%
from atac_rna_data_processing.io.mutation import read_rsid_parallel, Mutations, MutationsInCellType
variants_rsid = read_rsid_parallel(hg38, 'myc_rsid.txt', 5)
#%%
normal_variants = pd.read_csv('/manitou/pmg/users/xf2217/gnomad/myc.tad.vcf.gz', sep='\t', comment='#', header=None)
normal_variants.columns = ['Chromosome', 'Start', 'RSID', 'Ref', 'Alt', 'Qual', 'Filter', 'Info']
normal_variants['End'] = normal_variants.Start
normal_variants['Start'] = normal_variants.Start-1
normal_variants = normal_variants[['Chromosome', 'Start', 'End', 'RSID', 'Ref', 'Alt', 'Qual', 'Filter', 'Info']]
normal_variants = normal_variants.query('Ref.str.len()==1 & Alt.str.len()==1')
normal_variants['AF'] = normal_variants.Info.transform(lambda x: float(re.findall(r'AF=([0-9e\-\.]+)', x)[0]))
normal_variants_df = normal_variants.copy().query('AF>0.01').drop_duplicates(subset='RSID').query('RSID!="." & RSID!="rs55705857"')
#%%
sys.path.append('/manitou/pmg/users/xf2217/get_model/')
from inference import InferenceModel
import torch
checkpoint_path = '/manitou/pmg/projects/resources/get_interpret/pretrain_finetune_natac_fetal_adult.pth'
inf_model = InferenceModel(checkpoint_path, 'cuda')

#%%
CellCollection = {}
CellMutCollection = {}
results = []
from glob import glob
for cell_id in tqdm(sorted(glob('/manitou/pmg/users/xf2217/Interpretation_all_hg38_allembed_v4_natac/*'))):
    # try:
    cell_id = os.path.basename(cell_id)
    cell_type = cell_type_annot_dict[cell_id]
    CellCollection[cell_type] = GETCellType(cell_id, GET_CONFIG)
    if pr(CellCollection[cell_type].peak_annot).join(pr(variants_rsid.df)).df.empty:
        results.append([cell_type, 1])
        continue

    cell_mut = MutationsInCellType(hg38, variants_rsid.df, CellCollection[cell_type])
    cell_mut.get_original_input(motif)
    cell_mut.get_altered_input(motif)
    CellMutCollection[cell_type] = cell_mut
    ref_exp, alt_exp = cell_mut.predict_expression('rs55705857', 'MYC', 100, 200, inf_model=inf_model)
    results.append([cell_type, alt_exp/ref_exp])
    # except:
        # results.append([cell_type,1])
        
results = pd.DataFrame(results, columns=['cell_type', 'fc'])
#%%
CellMutCollection = {}
from glob import glob
for cell_id in tqdm(CellCollection):
    # try:
    print(cell_id)
    if pr(CellCollection[cell_id].peak_annot).join(pr(variants_rsid.df)).df.empty:
        results.append([cell_type, 1])
        continue

    cell_mut = MutationsInCellType(hg38, variants_rsid.df, CellCollection[cell_id])
    cell_mut.get_original_input(motif)
    cell_mut.get_altered_input(motif)
    CellMutCollection[cell_id] = cell_mut
#%%
results = []
from glob import glob
for cell_mut_id in tqdm(CellMutCollection):
    cell_mut = CellMutCollection[cell_mut_id]
    for rsid in cell_mut.mut.RSID.values:
        ref_exp, alt_exp = cell_mut.predict_expression(rsid, 'MYC', 100, 200, inf_model=inf_model)
        results.append([cell_mut_id, alt_exp/ref_exp, rsid])
        
results = pd.DataFrame(results, columns=['cell_type', 'fc', 'rsid'])

#%%
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

def process_cell_mutation(cell_mut_id):
    cell_mut = CellMutCollection[cell_mut_id]
    temp_results = []
    for rsid in cell_mut.mut.RSID.values:
        ref_exp, alt_exp = cell_mut.predict_expression(rsid, 'MYC', 100, 200, inf_model=inf_model)
        temp_results.append([cell_mut_id, alt_exp/ref_exp, rsid])
    return temp_results

results = []

# Using ProcessPoolExecutor to parallelize the loop
with ProcessPoolExecutor() as executor:
    # Mapping the function over the CellMutCollection keys
    for result in executor.map(process_cell_mutation, CellMutCollection.keys()):
        results.extend(result)

# Convert to DataFrame
results_df = pd.DataFrame(results, columns=['cell_type', 'fc', 'rsid'])
#%%
#violin plot of fc, highlighting adult olig and fetal astrocyte
fig, ax = plt.subplots(figsize=(4,3))
results['brain-related'] = results.cell_type.transform(lambda x: 'Oligodendrocyte' if 'ligoden' in x else 'Astrocyte' if 'strocyte' in x else 'Neuron' if 'euron' in x else 'Other')
sns.violinplot(x='brain-related', y='fc', data=results, ax=ax, palette=['#1f77b4', '#ff7f0e'])
# rotate xticklabels
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')

#%%
# extract AF from info, with scientific notation

#%%
# normal_variants = Mutations(hg38, normal_variants_df)
CellMut = MutationsInCellType(hg38, variants_rsid.df, FetalAst1)
# CellCommon = MutationsInCellType(hg38, normal_variants_df, AdultOligP)
#%%
CellMut.get_original_input(motif)
CellMut.get_altered_input(motif)
#%%
# CellCommon.get_original_input(motif)
# CellCommon.get_altered_input(motif)
#%%
import torch

def predict_expression(rsid, gene, CellMut, center, N, inf_model=inf_model):
    """
    Calculate expression predictions for original and altered cell states based on rsid and gene.

    Args:
    rsid (str): Reference SNP ID.
    gene (str): Gene name.
    CellMut (object): An instance of the CellMut class, containing data and methods for cell mutations.
    center (int): Center position for slicing the data matrix.
    N (int): The size of the slice from the data matrix.

    Returns:
    tuple: A tuple containing expression predictions for the original and altered states.
    """
    # Calculate new motif
    ref = CellMut.mut.df.query('RSID==@rsid').Ref.values[0]
    alt = CellMut.mut.df.query('RSID==@rsid').Alt.values[0]
    new_motif = (CellMut.Alt_input.loc[f'{rsid}_{alt}'].sparse.to_dense().values + 0.01) / \
                (CellMut.Ref_input.loc[f'{rsid}_{ref}'].sparse.to_dense().values + 0.01)

    # Determine start and end indices based on the gene TSS
    gene_tss_info = CellMut.celltype.get_gene_tss(gene)[0]
    start = gene_tss_info.peak_id - center
    end = start + N

    # Get strand information
    strand_idx = gene_tss_info.strand

    # Process original matrix
    original_matrix = CellMut.celltype.input_all[start:end].toarray()
    atac = original_matrix[:, 282].copy()
    original_matrix[:, 282] = 1

    # Process altered matrix
    idx_altered = CellMut.mut.df.query('RSID==@rsid').values[0][0] - start
    print(idx_altered)
    altered_matrix = original_matrix.copy()
    altered_matrix[:, 0:282] = new_motif * altered_matrix[:, 0:282]

    # Create tensors for prediction
    original = torch.Tensor(original_matrix).unsqueeze(0).to(inf_model.device)
    altered = torch.Tensor(altered_matrix).unsqueeze(0).to(inf_model.device)
    seq = torch.randn(1, N, 283, 4).to(inf_model.device)  # Dummy seq data
    tss_mask = torch.ones(1, N).to(inf_model.device)  # Dummy TSS mask
    ctcf_pos = torch.ones(1, N).to(inf_model.device)  # Dummy CTCF positions

    # Predict expression
    _, original_exp = inf_model.predict(original, seq, tss_mask, ctcf_pos)
    _, altered_exp = inf_model.predict(altered, seq, tss_mask, ctcf_pos)

    # Calculate and return the expression predictions
    original_pred = 10 ** (original_exp[0, center, strand_idx].item()) - 1
    altered_pred = 10 ** (altered_exp[0, center, strand_idx].item()) - 1

    return original_pred, altered_pred

#%%
predict_expression('rs55705857', 'MYC', CellMut, 100, 200, inf_model=inf_model)

#%%
common_result = []
for rsid in tqdm(CellCommon.mut.RSID.values):
    # try:
    original_pred, altered_pred = predict_expression(rsid.split('_')[0], 'MYC', CellCommon, 100, 200, inf_model=inf_model)
    common_result.append([original_pred, altered_pred, rsid.split('_')[0]])
    # except:
        # continue
#%%
mut_result = []
for rsid in tqdm(CellMut.mut.RSID.values):
    original_pred, altered_pred = predict_expression(rsid.split('_')[0], 'MYC', CellMut, 100, 200, inf_model=inf_model)
    mut_result.append([original_pred, altered_pred, rsid])
#%%
common_result = pd.DataFrame(common_result, columns=['original', 'altered', 'rsid'])
common_result['diff'] = common_result.altered / common_result.original

mut_result = pd.DataFrame(mut_result, columns=['original', 'altered', 'rsid'])
mut_result['diff'] = mut_result.altered / mut_result.original
mut_result['group'] = 'mut'
common_result['group'] = 'common'
#%%
results = pd.concat([mut_result, common_result])
# plot violin plot of diff
fig, ax = plt.subplots(figsize=(4,3))   
sns.violinplot(x='group', y='diff', data=results, ax=ax, palette=['#1f77b4', '#ff7f0e'], cut=0)
ax.set_xticklabels(['rs55705857', 'GnomAD AF>0.01'])
ax.set_xlabel('')
ax.set_ylabel('MYC Expression TPM fold change')
# add p-value annotation of mut to common by sum(common>mut)/len(common)
p = sum(common_result['diff']>mut_result['diff'].values[0])/len(common_result)
plt.text(s=f'Adult Oligodendrocyte Precursor P={p:.2f}', x=0.5, y=1.1, ha='center', va='bottom', transform=ax.transAxes)

plt.tight_layout()


#%%
# sns.scatterplot(np.arange(N), 10**altered_exp[0,:,1]-1, color='b', label='Altered')
# sns.scatterplot(np.arange(N), 10**original_exp[0,:,1]-1, color='r', label='Original')
# # plot atac in same plot by right axis
# ax2 = plt.twinx()
# sns.lineplot(np.arange(N), atac, color='grey', ax=ax2, label='ATAC')

#%%

#%%
variants_ref = pd.read_csv('glioma_variants.txt', sep='\t').set_index('ID').Ref.to_dict() 
variants_alt = pd.read_csv('glioma_variants.txt', sep='\t').set_index('ID').Alt.to_dict()
#%%
variants_ld = pd.read_csv('glioma_variants.txt', sep='\t')
ld = {}
lead_snp = ''
for i, row in variants_ld.iterrows():
    if row['Variant/LD'] == 'variant':
        lead_snp = row['ID']
        ld[row['ID']] = row['ID']
    else:
        ld[row['ID']] = lead_snp



#%%
variants_rsid = variants_rsid.df
# fix rsid table
variants_rsid['Ref'] = variants_rsid.RSID.map(variants_ref)
variants_rsid['Alt'] = variants_rsid.RSID.map(variants_alt)
#%%
variants_rsid = variants_rsid.dropna()
#%%
variants_rsid = Mutations(hg38, variants_rsid)
#%%
variants_rsid.collect_ref_sequence()
variants_rsid.collect_alt_sequence()
#%%
motif_diff = variants_rsid.get_motif_diff(motif)
#%%
motif_diff_df = pd.DataFrame((motif_diff['Alt'].values-motif_diff['Ref'].values), index=variants_rsid.df.RSID.values, columns=motif.cluster_names)
#%%
motif_diff_df.to_csv('motif_diff_df.csv')
#%%
variants_rsid_df = variants_rsid.df
def get_nearby_genes(variant, cell, distance=2000000):
    chrom = variant['Chromosome']
    pos = variant['Start']
    start = pos-distance
    end = pos+distance
    genes = cell.gene_annot.query('Chromosome==@chrom & Start>@start & Start<@end')
    return ','.join(np.unique(genes.gene_name.values))

variants_rsid_df['gene'] = variants_rsid_df.apply(lambda x: get_nearby_genes(x, AdultOlig), axis=1)
#%%
normal_variants_df['gene'] = normal_variants_df.apply(lambda x: get_nearby_genes(x, AdultAst1), axis=1)
#%%

gene_counts = variants_rsid_df.gene.str.split(',').explode().value_counts().reset_index()
#%%
def get_variant_score(motif_diff_score, variant, gene, cell):
    motif_importance = cell.get_gene_jacobian_summary(gene, 'motif')[0:-1]
    diff = motif_diff_score.copy().values
    diff[(diff<0) & (diff>-10)] = 0
    diff[(diff<0) & (diff<-10)] = -1
    diff[(diff>0) & (diff<10)] = 0
    diff[(diff>0) & (diff>10)] = 1
    
    combined_score = diff*motif_importance.values
    combined_score = pd.Series(combined_score, index=motif_diff_score.index.values).sort_values()
    combined_score = pd.DataFrame(combined_score, columns=['score'])
    combined_score['gene'] = gene
    combined_score['variant'] = variant.RSID
    try:
        combined_score['ld'] = ld[variant.RSID]
    except:
        combined_score['ld'] = variant.RSID
    combined_score['chrom'] = variant.Chromosome
    combined_score['pos'] = variant.Start
    combined_score['ref'] = variant.Ref
    combined_score['alt'] = variant.Alt
    combined_score['celltype'] = cell_type_annot_dict[cell.celltype]
    return combined_score

get_variant_score(motif_diff_df.loc["rs55705857"], variants_rsid_df.query('RSID=="rs55705857"').iloc[0], 'EGFR', AdultOlig).sort_values('score')
#%%
from tqdm import tqdm
normal_scores = pd.DataFrame()
for i, row in tqdm(normal_variants_df.iterrows()):
    for gene in row.gene.split(','):
        for cell in [AdultAst1]:
            try:
                score = get_variant_score(normal_motif_diff_df.loc[row.RSID], row, gene, cell)
                normal_scores = pd.concat([normal_scores, score])
            except:
                continue

#%%
scores = pd.DataFrame()
for i, row in tqdm(variants_rsid_df.iterrows()):
    for gene in row.gene.split(','):
        for cell in [AdultOlig]:
            try:
                score = get_variant_score(motif_diff_df.loc[row.RSID], row, gene, cell)
                scores = pd.concat([scores, score])
            except:
                continue
#%%
scores.reset_index().to_feather('glioma_scores.olig.feather')
scores.reset_index().to_csv('glioma_scores.olig.csv')
#%%
scores = pd.read_feather('glioma_scores.feather')
scores_olig = pd.read_feather('glioma_scores.olig.feather')
#%%
scores = pd.concat([scores, scores_olig])
#%%
scores.query('variant=="rs55705857" & gene=="MYC"')

#%%
scores_tested = pd.DataFrame()
for gene in tqdm(normal_scores.gene.unique()):
    normal_scores_gene = normal_scores.query('gene==@gene').reset_index().query('score!=0 & index=="POU/3"')
    null = normal_scores_gene.score.values
    print(null)
    # add two tail p-value to scores_tp53 based on null
    scores_gene = scores.query('gene==@gene').query('score!=0')
    try:
        scores_gene['p_right'] = scores_gene.score.transform(lambda x: (sum(null>=x))/(len(null)))
        scores_gene['p_left'] = scores_gene.score.transform(lambda x: (sum(null<=x))/(len(null)))
        scores_tested = pd.concat([scores_tested, scores_gene])
    except:
        pass


scores_tested['p'] = scores_tested[['p_right', 'p_left']].min(1)

#%%
scores_tested.query('p<0.025 & celltype=="Astrocyte 1"').sort_values('p')






#%%
# ld = [
# "rs4977756",
# "rs2383205",
# "rs2184061",
# "rs1537378",
# "rs8181050",
# "rs1333039",
# "rs4977755",
# "rs10965223",
# "rs10965224",
# "rs10811648",
# "rs10811649",
# "rs10811651",
# ]
# scores.query('gene=="EGFR"').groupby('variant').score.sum().sort_values()
# plot scatter with variant name annotation, x is np.arange(len(variants)), y is score
def plot_gene_variant_sum_effect(gene, scores):
    # highlight ld variants in red
    df = scores.query('gene==@gene & celltype=="Fetal Astrocyte 1"').groupby('variant').score.sum().sort_values()
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.scatterplot(x=np.arange(len(df)), y=df.values, ax=ax, hue = df.values, palette='RdBu_r', legend=False, hue_norm=(-0.001, 0.001))
    # add y=0 line
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    for i, row in df.reset_index().iterrows():
        # only for top and bottom 5
        if i<5 or i>len(df)-5:
        # if row.variant in ["rs4977756", "rs2383205","rs10811648", "rs10811649",]:
            plt.text(s=row.variant, x=i+0.7, y=row.score, ha='center', va='bottom', alpha=1)

    # remove xticks
    # xlim to 0,20
    # plt.xlim(-1,20)
    plt.xticks([])
    # y label:
    plt.ylabel('Impact score on expression')
    # plt.title('Variants linked to rs4977756', y=1.1)
    plt.tight_layout()
    return fig, ax


#%%


#%%
def query_variant_region(variant, gene, cell):
    chrom = variant['Chromosome']
    pos = variant['Start']
    df = cell.peak_annot.iloc[cell.get_gene_jacobian_summary(gene, 'region')['index'].values].query('Start<@pos & End>@pos & Chromosome==@chrom')
    if len(df) == 0:
        pass 
    else:
        print(df, variant.RSID, cell.celltype)
        return variant.RSID


variant_in_peak = []
for i, row in variants_rsid_df.query('Chromosome=="chr7"').iterrows():
    for cell in [FetalAst1, FetalAst2, FetalAst3, FetalAst4, AdultAst1, AdultOlig]:
        variant_in_peak.append(query_variant_region(row, 'EGFR', cell))

# remove None
variant_in_peak = np.unique([i for i in variant_in_peak if i is not None])

# query_variant_region(variants_rsid_df.query('RSID=="rs2184061"').iloc[0], 'CDKN2A', AdultOlig)
#%%
scores['ld'] = scores.variant.map(ld)
scores['zscore'] = scores.score.transform(lambda x: (x-x.mean())/x.std())
scores = scores.query('variant!="rs140317985"')
scores_in_peak = scores.query('variant.isin(@variant_in_peak)').reset_index().sort_values('score')
scores_in_peak['ld'] = scores_in_peak.variant.map(ld)
scores_in_peak['zscore'] = scores_in_peak.score.transform(lambda x: (x-x.mean())/x.std())
plot_gene_variant_sum_effect('EGFR', scores_in_peak)

#%%
#%%
motif_diff_df.loc['rs11979406'].sort_values()
#%%
AdultAst1.get_gene_jacobian('EGFR', 'region')[2].data.query('Start==55086065').iloc[0,4:].T.sort_values()['NFI/1']
#%%
AdultAst1.gene_jacobian_summary('EGFR', 'region').query('Start==55086065')#.iloc[0,4:].T.sort_values()
#%%
FetalAst1.gene_jacobian_summary('EGFR').sort_values().tail(20)
#%%
FetalAst2.gene_annot.query('gene_name.str.startswith("HOX")').sort_values('pred')
#%%
variants.query('gene.str.contains("CDKN2A") & (Motif_changed.str.contains("Pax") | Motif_changed.str.contains("Mzf"))')
#%%
causal = FetalAst1.gene_by_motif.get_causal()
plot_comm(get_subnet(preprocess_net(causal.copy(), 0.09), 'NFI/1'), figsize=(6,6), title="")
#%%
pax_causal = get_subnet(preprocess_net(causal.copy(), 0.1), 'PAX/2')
causal_top_edges = pd.read_csv('causal_top_pair_per_celltype_df.csv')

# replace node size and name in pax_causal with gene expression in causal_top_edges (columns:'source', 'target', 'score', 'exp_source', 'exp_target', 'exp_overall', 'celltype', 'exp_source_gene', 'exp_target_gene')
pax_causal = pax_causal.copy()
#%%
# plot expression across cell types
# add gene name annotation for each bar using exp_source_gene
fix, ax = plt.subplots(figsize=(3,2))
exp_df = causal_top_edges[['source', 'exp_source', 'exp_source_gene', 'celltype']].drop_duplicates().query('source=="PAX/2"').sort_values('exp_source', ascending=True)
# reduce space between bars
exp_df.plot(x='celltype',y='exp_source', kind='barh', ylabel='', xlabel=r'$\log_{10}$TPM', legend=False, color='#1f77b4', ax=ax, alpha=0.5, width=0.8)
for i, row in exp_df.reset_index().iterrows():
    # ConversionError: Failed to convert value(s) to axis units: 'Fetal Thymocyte'
    plt.text(s=row.exp_source_gene, y=i, x=0.6, ha='right', va='center', color='#FFFFFF', alpha=1)
#%%
def plot_comm(G, figsize=(10, 10), title='Network structure', savefig=False):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # Set community color for internal edges
    external = [(v, w) for v, w in G.edges if G.edges[v, w]['community'] == 0]
    internal = [(v, w) for v, w in G.edges if G.edges[v, w]['community'] > 0]
    internal_color = ["#ef8a62" if (G.edges[e]['weight'] > 0) else "#67a9cf" for e in internal]
    external_color = ["#ef8a62" if (G.edges[e]['weight'] > 0) else "#67a9cf" for e in external]
    node_color = [get_color(G.nodes[v]['community']) for v in G.nodes]

    weights = [np.absolute(G[u][v]['weight'])*20 for u,v in G.edges]
    # external edges
    nx.draw(
        G,
        pos=nx.nx_agraph.graphviz_layout(G, 'neato'),
        node_size=0,
        edgelist=external,
        width = weights,
        ax=ax,
        edge_color=external_color,
        node_color=node_color,
        with_labels=True)
    # internal edges
    nx.draw(
        G, 
        pos=nx.nx_agraph.graphviz_layout(G, 'neato'),
        node_size=100,
        edgelist=internal,
        ax=ax,
        width = weights,
        edge_color=internal_color,
        node_color=node_color,
        with_labels=True)
    ax.set_title(title)
    plt.tight_layout()
    # add padding in the margins
    plt.margins(0.3, 0.3)
    if savefig:
        plt.savefig(os.path.join(savefig, title + '.png'), dpi=300)
    plt.show()
    plt.close()
    
# plot_comm(pax_causal, figsize=(3,3), title="")
#%%
# read RNAseq of B_ALL
# sample annotation /manitou/pmg/users/xf2217/interpret_natac/Lineage_ATAC_B.csv
sample_annot = pd.read_csv('/manitou/pmg/users/xf2217/interpret_natac/Lineage_ATAC_B.csv')

# rna fpkm files in /manitou/pmg/users/xf2217/interpret_natac/B_ALL_RNAseq/*all_fpkm.txt
# read all fpkm files and merge into one dataframe
fpkm_files = [os.path.join('/manitou/pmg/users/xf2217/interpret_natac/B_ALL_RNAseq/', i) for i in os.listdir('/manitou/pmg/users/xf2217/interpret_natac/B_ALL_RNAseq/') if i.endswith('all_fpkm.txt') and not i.startswith('6329')]

fpkm = pd.concat([pd.read_csv(i, sep='\t', index_col=0).iloc[:,-1].rename(
    os.path.basename(i).split('.')[0]) for i in fpkm_files], axis=1)
fpkm.columns = [sample_annot.set_index('RNAseq')['Pair ID'].to_dict()[c] for c in fpkm.columns]

# read first 7 columns of one fpkm file as gene annotation
gene_annot = pd.read_csv(fpkm_files[0], sep='\t', index_col=0).iloc[:,:6]

fpkm_files_6239 = [os.path.join('/manitou/pmg/users/xf2217/interpret_natac/B_ALL_RNAseq/', i) for i in os.listdir('/manitou/pmg/users/xf2217/interpret_natac/B_ALL_RNAseq/') if i.endswith('all_fpkm.txt') and i.startswith('6329')]

# fpkm_6239 = pd.concat([pd.read_csv(i, sep='\t', index_col=0).iloc[:,-1].rename(
    # os.path.basename(i).split('.')[0]) for i in fpkm_files_6239], axis=1)
# fpkm_6239.columns = [sample_annot.set_index('RNAseq')['Pair ID'].to_dict()[c] for c in fpkm_6239.columns]
# gene_annot_6239 = pd.read_csv(fpkm_files_6239[0], sep='\t', index_col=0).iloc[:,:6]

# pd merge the two fpkm dataframes basead on index
# fpkm = pd.merge(fpkm, fpkm_6239, left_index=True, right_index=True)
# merge with gene annotation
# gene_annot = pd.concat([gene_annot, gene_annot_6239], axis=0).drop_duplicates()

# join gene_annot with fpkm
fpkm = pd.merge(gene_annot, fpkm, left_index=True, right_index=True)

sample_to_keep = ['MH3266 D', '26557 D', 'MH1067 D', 'MH827 D',
       'MH2955 D', 'MH2559 D', '27424 D', '26639 D', '25471 D', '27250 D',
        'MH410 D', 'MH2738 D', '21691 D', 'MH1550 D']
#%%
from gprofiler import GProfiler
gp = GProfiler(return_dataframe=True)

def get_tf_pathway(tf1, tf2, cell, filter_str='term_size<1000 & term_size>100'):
    df = cell.gene_annot.query('pred>0')
    tf1_genes = cell.gene_annot.iloc[cell.gene_by_motif.data[tf1].sort_values().tail(10000).index.values]
    tf2_genes = cell.gene_annot.iloc[cell.gene_by_motif.data[tf2].sort_values().tail(10000).index.values]
    intersect_genes = tf1_genes.merge(tf2_genes, on='gene_name').gene_name.unique()
    tf1_genes = tf1_genes.gene_name.unique()
    tf2_genes = tf2_genes.gene_name.unique()
    background = df.query('pred>0').gene_name.unique()
    # keep only specific genes by remove intersect genes
    tf1_genes = np.setdiff1d(tf1_genes, intersect_genes)
    tf2_genes = np.setdiff1d(tf2_genes, intersect_genes)
    go_tf1 = gp.profile(organism='hsapiens', query=list(tf1_genes), user_threshold=0.05, no_evidences=False, background=list(background))
    go_tf2 = gp.profile(organism='hsapiens', query=list(tf2_genes), user_threshold=0.05, no_evidences=False, background=list(background))
    go_intersect = gp.profile(organism='hsapiens', query=list(intersect_genes), user_threshold=0.05, no_evidences=False, background=list(background))
    go_tf1_filtered = go_tf1.query(filter_str)
    go_tf2_filtered = go_tf2.query(filter_str)
    go_intersect_filtered = go_intersect.query(filter_str)

    return tf1_genes, tf2_genes, intersect_genes, go_tf1_filtered, go_tf2_filtered, go_intersect_filtered

tf1_genes, tf2_genes, intersect_genes, go_tf1_filtered, go_tf2_filtered, go_intersect_filtered = get_tf_pathway('PAX/2', 'NR/3', Bcell, filter_str = 'term_size < 5000')
#%%


#%%
# venn plot of 2856, 2192, 881
from matplotlib_venn import venn2
fig, ax = plt.subplots(figsize=(3,3))
venn2([set(tf1_genes).union(set(intersect_genes)), set(tf2_genes).union(set(intersect_genes))], alpha=0.5,  set_labels=['PAX/2', 'NR/3'], set_colors=['#1f77b4', '#ff7f0e'], ax=ax)
#%%
# separate fpkm to columns end with R and D respectively
intersect_genes_fpkm = fpkm.query('GeneName.isin(@intersect_genes)').iloc[:, fpkm.columns.isin(sample_to_keep)].mean(0)
tf1_genes_fpkm = fpkm.query('GeneName.isin(@tf1_genes)').iloc[:, fpkm.columns.isin(sample_to_keep)].mean(0)
tf2_genes_fpkm = fpkm.query('GeneName.isin(@tf2_genes)').iloc[:, fpkm.columns.isin(sample_to_keep)].mean(0)
# random_pax5 = Bcell.gene_annot.iloc[Bcell.gene_by_motif.data['PAX/2'].abs().sort_values().head(10000).index.values].gene_name.unique()
# random_rora = Bcell.gene_annot.iloc[Bcell.gene_by_motif.data['NR/3'].abs().sort_values().head(10000).index.values].gene_name.unique()
random_intersect = np.concatenate((tf1_genes, tf2_genes))
fpkm_random = fpkm.query('~GeneName.isin(@random_intersect)')
#%%
pax5_fpkm = fpkm.query('GeneName=="PAX5"').iloc[:, fpkm.columns.isin(sample_to_keep)].mean(0)
random_tf1_size_genes_fpkm_corr = np.stack([pax5_fpkm.corr(fpkm.sample(len(tf1_genes)).iloc[:, fpkm.columns.isin(sample_to_keep)].mean(0), method='spearman') for i in range(10000)])
random_tf2_size_genes_fpkm_corr = np.stack([pax5_fpkm.corr(fpkm.sample(len(tf2_genes)).iloc[:, fpkm.columns.isin(sample_to_keep)].mean(0), method='spearman') for i in range(10000)])
random_intersect_size_genes_fpkm_corr = np.stack([pax5_fpkm.corr(fpkm.sample(len(intersect_genes)).iloc[:, fpkm.columns.isin(sample_to_keep)].mean(0), method='spearman') for i in range(10000)])
#%%

# plot scatterplot of each gene vs PAX5
fig, ax = plt.subplots(1, 3, figsize=(9,3))
sns.scatterplot(x=pax5_fpkm, y=tf1_genes_fpkm, ax=ax[0], color='#1f77b4', alpha=0.5)
sns.scatterplot(x=pax5_fpkm, y=tf2_genes_fpkm, ax=ax[1], color='#ff7f0e',alpha=0.5)
sns.scatterplot(x=pax5_fpkm, y=intersect_genes_fpkm, ax=ax[2], color='#E9DDCF', alpha=0.9)
# annotate the spearman correlation on each plot
ax[0].annotate(f"Correlation: {pax5_fpkm.corr(tf1_genes_fpkm, method='pearson'):.2f}, P={sum(random_tf1_size_genes_fpkm_corr>pax5_fpkm.corr(tf1_genes_fpkm, method='pearson'))/10000:.2f}", xy=(0.05, 1.1), xycoords='axes fraction')
ax[0].annotate(f"Random: {random_tf1_size_genes_fpkm_corr.mean():.2f}", xy=(0.05, 1.0), xycoords='axes fraction')
ax[1].annotate(f"Correlation: {pax5_fpkm.corr(tf2_genes_fpkm, method='pearson'):.2f}, P={sum(random_tf2_size_genes_fpkm_corr>pax5_fpkm.corr(tf2_genes_fpkm, method='pearson'))/10000:.2f}", xy=(0.05, 1.1), xycoords='axes fraction')
ax[1].annotate(f"Random: {random_tf2_size_genes_fpkm_corr.mean():.2f}", xy=(0.05, 1.0), xycoords='axes fraction')
ax[2].annotate(f"Correlation: {pax5_fpkm.corr(intersect_genes_fpkm, method='pearson'):.2f}, P={sum(random_intersect_size_genes_fpkm_corr>pax5_fpkm.corr(intersect_genes_fpkm, method='pearson'))/10000:.2f}", xy=(0.05, 1.1), xycoords='axes fraction')
ax[2].annotate(f"Random: {random_intersect_size_genes_fpkm_corr.mean():.2f}", xy=(0.05, 1.0), xycoords='axes fraction')
ax[0].set_ylabel('Target gene expression\nPAX/2 specific')
ax[1].set_ylabel('NR/3 specific')
ax[2].set_ylabel('PAX/2+NR/3 common')
ax[0].set_xlabel('PAX5 expression')
ax[1].set_xlabel('PAX5 expression')
ax[2].set_xlabel('PAX5 expression')
plt.tight_layout()
#%%
# random_tf1_size_genes_fpkm_corr = np.stack([pax5_fpkm.corr(fpkm_random.sample(len(tf1_genes)).iloc[:, fpkm.columns.isin(sample_to_keep)].mean(0), method='spearman') for i in range(10000)])
# random_tf2_size_genes_fpkm_corr = np.stack([pax5_fpkm.corr(fpkm_random.sample(len(tf2_genes)).iloc[:, fpkm.columns.isin(sample_to_keep)].mean(0), method='spearman') for i in range(10000)])
# random_intersect_size_genes_fpkm_corr = np.stack([pax5_fpkm.corr(fpkm_random.sample(len(intersect_genes)).iloc[:, fpkm.columns.isin(sample_to_keep)].mean(0), method='spearman') for i in range(10000)])
pax5_fpkm = fpkm.query('GeneName=="RORA"').iloc[:, fpkm.columns.isin(sample_to_keep)].mean(0)
# plot scatterplot of each gene vs PAX5
fig, ax = plt.subplots(1, 3, figsize=(9,3))
sns.scatterplot(x=pax5_fpkm, y=tf1_genes_fpkm, ax=ax[0], color='#1f77b4', alpha=0.5)
sns.scatterplot(x=pax5_fpkm, y=tf2_genes_fpkm, ax=ax[1], color='#ff7f0e',alpha=0.5)
sns.scatterplot(x=pax5_fpkm, y=intersect_genes_fpkm, ax=ax[2], color='#E9DDCF', alpha=0.9)
# annotate the spearman correlation on each plot
ax[0].annotate(f"Correlation: {pax5_fpkm.corr(tf1_genes_fpkm, method='pearson'):.2f}, P={sum(random_tf1_size_genes_fpkm_corr>pax5_fpkm.corr(tf1_genes_fpkm, method='pearson'))/10000:.2f}", xy=(0.05, 1.1), xycoords='axes fraction')
ax[0].annotate(f"Random: {random_tf1_size_genes_fpkm_corr.mean():.2f}", xy=(0.05, 1.0), xycoords='axes fraction')
ax[1].annotate(f"Correlation: {pax5_fpkm.corr(tf2_genes_fpkm, method='pearson'):.2f}, P={sum(random_tf2_size_genes_fpkm_corr>pax5_fpkm.corr(tf2_genes_fpkm, method='pearson'))/10000:.2f}", xy=(0.05, 1.1), xycoords='axes fraction')
ax[1].annotate(f"Random: {random_tf2_size_genes_fpkm_corr.mean():.2f}", xy=(0.05, 1.0), xycoords='axes fraction')
ax[2].annotate(f"Correlation: {pax5_fpkm.corr(intersect_genes_fpkm, method='pearson'):.2f}, P={sum(random_intersect_size_genes_fpkm_corr>pax5_fpkm.corr(intersect_genes_fpkm, method='pearson'))/10000:.2f}", xy=(0.05, 1.1), xycoords='axes fraction')
ax[2].annotate(f"Random: {random_intersect_size_genes_fpkm_corr.mean():.2f}", xy=(0.05, 1.0), xycoords='axes fraction')
ax[0].set_ylabel('Target gene expression\nPAX/2 specific')
ax[1].set_ylabel('NR/3 specific')
ax[2].set_ylabel('PAX/2+NR/3 common')
ax[0].set_xlabel('RORA expression')
ax[1].set_xlabel('RORA expression')
ax[2].set_xlabel('RORA expression')
plt.tight_layout()

#%%
pax5_fpkm = fpkm.query('GeneName=="RARA"').iloc[:, fpkm.columns.isin(sample_to_keep)].mean(0)
# plot scatterplot of each gene vs PAX5
fig, ax = plt.subplots(1, 3, figsize=(9,3))
sns.scatterplot(x=pax5_fpkm, y=tf1_genes_fpkm, ax=ax[0], color='#1f77b4', alpha=0.5)
sns.scatterplot(x=pax5_fpkm, y=tf2_genes_fpkm, ax=ax[1], color='#ff7f0e',alpha=0.5)
sns.scatterplot(x=pax5_fpkm, y=intersect_genes_fpkm, ax=ax[2], color='#E9DDCF', alpha=0.9)
# annotate the spearman correlation on each plot
ax[0].annotate(f"Correlation: {pax5_fpkm.corr(tf1_genes_fpkm, method='pearson'):.2f}, P={sum(random_tf1_size_genes_fpkm_corr>pax5_fpkm.corr(tf1_genes_fpkm, method='pearson'))/10000:.2f}", xy=(0.05, 1.1), xycoords='axes fraction')
ax[0].annotate(f"Random: {random_tf1_size_genes_fpkm_corr.mean():.2f}", xy=(0.05, 1.0), xycoords='axes fraction')
ax[1].annotate(f"Correlation: {pax5_fpkm.corr(tf2_genes_fpkm, method='pearson'):.2f}, P={sum(random_tf2_size_genes_fpkm_corr>pax5_fpkm.corr(tf2_genes_fpkm, method='pearson'))/10000:.2f}", xy=(0.05, 1.1), xycoords='axes fraction')
ax[1].annotate(f"Random: {random_tf2_size_genes_fpkm_corr.mean():.2f}", xy=(0.05, 1.0), xycoords='axes fraction')
ax[2].annotate(f"Correlation: {pax5_fpkm.corr(intersect_genes_fpkm, method='pearson'):.2f}, P={sum(random_intersect_size_genes_fpkm_corr>pax5_fpkm.corr(intersect_genes_fpkm, method='pearson'))/10000:.2f}", xy=(0.05, 1.1), xycoords='axes fraction')
ax[2].annotate(f"Random: {random_intersect_size_genes_fpkm_corr.mean():.2f}", xy=(0.05, 1.0), xycoords='axes fraction')
ax[0].set_ylabel('Target gene expression\nPAX/2 specific')
ax[1].set_ylabel('NR/3 specific')
ax[2].set_ylabel('PAX/2+NR/3 common')
ax[0].set_xlabel('RARA expression')
ax[1].set_xlabel('RARA expression')
ax[2].set_xlabel('RARA expression')
plt.tight_layout()
#%%
pax5_fpkm = fpkm.query('GeneName=="NR4A1"').iloc[:, fpkm.columns.isin(sample_to_keep)].mean(0)
# plot scatterplot of each gene vs PAX5
fig, ax = plt.subplots(1, 3, figsize=(9,3))
sns.scatterplot(x=pax5_fpkm, y=tf1_genes_fpkm, ax=ax[0], color='#1f77b4', alpha=0.5)
sns.scatterplot(x=pax5_fpkm, y=tf2_genes_fpkm, ax=ax[1], color='#ff7f0e',alpha=0.5)
sns.scatterplot(x=pax5_fpkm, y=intersect_genes_fpkm, ax=ax[2], color='#E9DDCF', alpha=0.9)
# annotate the spearman correlation on each plot
ax[0].annotate(f"Correlation: {pax5_fpkm.corr(tf1_genes_fpkm, method='pearson'):.2f}, P={sum(random_tf1_size_genes_fpkm_corr>pax5_fpkm.corr(tf1_genes_fpkm, method='pearson'))/10000:.2f}", xy=(0.05, 1.1), xycoords='axes fraction')
ax[0].annotate(f"Random: {random_tf1_size_genes_fpkm_corr.mean():.2f}", xy=(0.05, 1.0), xycoords='axes fraction')
ax[1].annotate(f"Correlation: {pax5_fpkm.corr(tf2_genes_fpkm, method='pearson'):.2f}, P={sum(random_tf2_size_genes_fpkm_corr>pax5_fpkm.corr(tf2_genes_fpkm, method='pearson'))/10000:.2f}", xy=(0.05, 1.1), xycoords='axes fraction')
ax[1].annotate(f"Random: {random_tf2_size_genes_fpkm_corr.mean():.2f}", xy=(0.05, 1.0), xycoords='axes fraction')
ax[2].annotate(f"Correlation: {pax5_fpkm.corr(intersect_genes_fpkm, method='pearson'):.2f}, P={sum(random_intersect_size_genes_fpkm_corr>pax5_fpkm.corr(intersect_genes_fpkm, method='pearson'))/10000:.2f}", xy=(0.05, 1.1), xycoords='axes fraction')
ax[2].annotate(f"Random: {random_intersect_size_genes_fpkm_corr.mean():.2f}", xy=(0.05, 1.0), xycoords='axes fraction')
ax[0].set_ylabel('Target gene expression\nPAX/2 specific')
ax[1].set_ylabel('NR/3 specific')
ax[2].set_ylabel('PAX/2+NR/3 common')
ax[0].set_xlabel('NR4A1 expression')
ax[1].set_xlabel('NR4A1 expression')
ax[2].set_xlabel('NR4A1 expression')
plt.tight_layout()

#%%
tf1_genes_stat, tf2_genes_stat, intersect_genes_stat, go_tf1_filtered_stat, go_tf2_filtered_stat, go_intersect_filtered_stat = get_tf_pathway('STAT/1', 'NR/19', Bcell)
#%%
#%%
# parse PAX5 gene list named as PAX5_G183_genelist.txt
# for each line, read as a list l, append to the result list r as r.append(l[0], l[1], l[2:])
# convert to a pd.DataFrame with columns=['set', 'source', 'genes']

with open('PAX5_G183_genelist.txt', 'r') as f:
    r = []
    for line in f.readlines():
        l = line.strip().split('\t')
        r.append([l[0], l[1], l[2:]])
    pax5_genelist = pd.DataFrame(r, columns=['set', 'source', 'genes'])
#%%
from scipy.stats import fisher_exact
def fisher_exact_test(set1, set2, background):
    # Create a contingency table
    set1 = set(set1)
    set2 = set(set2)
    background = set(background)
    contingency_table = [
        [len(set1.intersection(set2)), len(set1.difference(set2))],
        [len(set2.difference(set1)), len(background.difference(set1.union(set2)))]
    ]

    # Perform Fisher's exact test
    odds_ratio, p_value = fisher_exact(contingency_table)
    # return p_value and common genes
    return p_value, set1.intersection(set2)

# for each line in pax5_genelist, compute Fisher exact test with intersect_genes by compute overlap between genes and intersect_genes

for i, row in pax5_genelist.iterrows():
    r = fisher_exact_test(row.genes, tf2_genes, Bcell.gene_annot.query('pred>0').gene_name.unique())
    pax5_genelist.loc[i, 'p_value'] = r[0]
    pax5_genelist.loc[i, 'intersections'] = ' '.join(r[1])
# compute fdr
from statsmodels.stats.multitest import multipletests
pax5_genelist['fdr'] = multipletests(pax5_genelist.p_value, method='fdr_bh')[1]
pax5_genelist['-log10p'] = -np.log10(pax5_genelist.p_value)
# remove underscore in set name
pax5_genelist['set'] = pax5_genelist['set'].str.replace('_', ' ')
# plot top 10
pax5_genelist.query('fdr<0.05').sort_values('p_value', ascending=False).tail(10).plot(x='set', y='-log10p', kind='barh', figsize=(3,2), legend=False, xlabel ='-log10 p-value', ylabel='Gene set', color='#ff7f0e', alpha=0.5, xlim=(0, 15))
#%%
for i, row in pax5_genelist.iterrows():
    r = fisher_exact_test(row.genes, intersect_genes, Bcell.gene_annot.query('pred>0').gene_name.unique())
    pax5_genelist.loc[i, 'p_value'] = r[0]
    pax5_genelist.loc[i, 'intersections'] = ' '.join(r[1])
# compute fdr
from statsmodels.stats.multitest import multipletests
pax5_genelist['fdr'] = multipletests(pax5_genelist.p_value, method='fdr_bh')[1]
pax5_genelist['-log10p'] = -np.log10(pax5_genelist.p_value)

# plot top 10
pax5_genelist.query('fdr<0.05').sort_values('p_value', ascending=False).tail(10).plot(x='set', y='-log10p', kind='barh', figsize=(3,2), legend=False, xlabel ='-log10 p-value', ylabel='Gene set', color='#E9DDCF', alpha=0.9, xlim=(0, 15))
#%%
for i, row in pax5_genelist.iterrows():
    r = fisher_exact_test(row.genes, tf1_genes, Bcell.gene_annot.query('pred>0').gene_name.unique())
    pax5_genelist.loc[i, 'p_value'] = r[0]
    pax5_genelist.loc[i, 'intersections'] = ' '.join(r[1])
# compute fdr
from statsmodels.stats.multitest import multipletests
pax5_genelist['fdr'] = multipletests(pax5_genelist.p_value, method='fdr_bh')[1]
pax5_genelist['-log10p'] = -np.log10(pax5_genelist.p_value)

# plot top 10
pax5_genelist.query('fdr<0.05').sort_values('p_value', ascending=False).tail(10).plot(x='set', y='-log10p', kind='barh', figsize=(3,2), legend=False, xlabel ='-log10 p-value', ylabel='Gene set', color='#1f77b4', alpha=0.5, xlim=(0, 15))
#%%
go_intersect_filtered.query('source=="GO:BP"')


#%%
tf1_genes, tf2_genes, intersect_genes, go_tf1_filtered, go_tf2_filtered, go_intersect_filtered = get_tf_pathway('PAX/2', 'NR/3', Bcell)
#%%
go_tf1_filtered.query('source=="GO:BP"').head(10)
#%%

go_tf2_filtered.query('source=="GO:BP"').head(10)

#%%
go_intersect_filtered.query('source=="GO:BP"').head(20)
#%%
# scatter plot comparing p-values of go_tf1_filtered and go_intersect_filtered, na filled as 0
# join then fill then plot
# -log10 pvalue first
go_tf1_filtered['p_value_log'] = -np.log10(go_tf1_filtered.p_value)
go_tf2_filtered['p_value_log'] = -np.log10(go_tf2_filtered.p_value)
go_intersect_filtered['p_value_log'] = -np.log10(go_intersect_filtered.p_value)
fig, ax = plt.subplots(figsize=(5,5))
pd.merge(go_tf1_filtered, go_intersect_filtered, on='native', how='outer').fillna(0).plot(x='p_value_log_x', y='p_value_log_y', kind='scatter', alpha=0.8, s=1, ax=ax)
ax.set_xlabel('PAX5')
ax.set_ylabel('PAX5+RORA')
#%%
go_intersect_filtered.query('source=="GO:BP" & term_size<500').sort_values('p_value', ascending=False).tail(10).plot(x='name', y='p_value_log', kind='barh', figsize=(3,2), legend=False, xlabel ='-log10 p-value', ylabel='Gene set', color='#E9DDCF', alpha=0.9,xlim=(0, 15))
#%%
go_tf1_filtered.query('source=="GO:BP" & term_size<500').sort_values('p_value', ascending=False).tail(10).plot(x='name', y='p_value_log', kind='barh', figsize=(3,2), legend=False, xlabel ='-log10 p-value', ylabel='Gene set', color='#1f77b4', alpha=0.5, xlim=(0, 20))
#%%
go_tf2_filtered.query('source=="GO:BP" & term_size<500').sort_values('p_value', ascending=False).tail(10).plot(x='name', y='p_value_log', kind='barh', figsize=(3,2), legend=False, xlabel ='-log10 p-value', ylabel='Gene set', color='#ff7f0e', alpha=0.5, xlim=(0, 20))
#%%
' '.join(np.unique(np.concatenate(pd.merge(go_tf1_filtered, go_intersect_filtered, on='native', how='outer').query('p_value_x.isna()').intersections_y.values)))

#%%
# sns.scatterplot(x='Ebox/CACCTG', y='Ebox/CAGATGG', data=Tcell.gene_by_motif.data, s=3, alpha=0.5)

tcell_vs_ery1 = pd.merge(pd.concat([Tcell.gene_annot,Tcell.gene_by_motif.data], axis=1).groupby('gene_name').mean(0), pd.concat([Ery1.gene_annot,Ery1.gene_by_motif.data], axis=1).groupby('gene_name').mean(0), left_on='gene_name', right_on='gene_name', how='outer', suffixes=['_Tcell', '_Ery1']).fillna(0)#.plot(x='Ebox/CACCTG_Tcell', y='Ebox/CACCTG_Ery1', kind='scatter', alpha=0.8, s=1)
tcell_vs_ery1 = tcell_vs_ery1.query('pred_Tcell-pred_Ery1>0.5 or pred_Tcell-pred_Ery1<-0.5')
fig, ax = plt.subplots(figsize=(5,5))
sns.scatterplot(x= tcell_vs_ery1['Ebox/CAGATGG_Ery1'], y= tcell_vs_ery1['Ebox/CACCTG_Ery1'], s=3, alpha=1, ax=ax, hue= tcell_vs_ery1['pred_Tcell']-tcell_vs_ery1['pred_Ery1'], palette='RdBu_r', legend=True)
ax.set_xlabel('Ebox/CAGATGG_Ery1')
ax.set_ylabel('Ebox/CACCTG_Ery1')
ax.set_xlim(-0.001, 0.001)
ax.set_ylim(-0.001, 0.001)
ax.set_title("Tcell vs Ery1")
# add x=0 y=0
ax.axvline(0, c='grey', alpha=0.5)
ax.axhline(0, c='grey', alpha=0.5)
# add correlation annotation

# Ebox/CACCTG: Pure TCF3, TCF12
# Ebox/CAGATGG: TAL1 + TCF3 

# E -> T

# TAL1 high -> low




# TAL1 expression up ->  TCF3

# TAL1 TCF3 - SIN3A
# TCF3 TCF12 - EP300

# Mutation -> MYB binding -> TAL1 Overexpression

# Tcell 177 in fetal: TAL1 is regulated by MYB/5 



# %%
