#%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from glob import glob
from sys import argv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix, load_npz
from scipy.stats import zscore
from sklearn.isotonic import spearmanr
from tqdm import tqdm
plt.style.use('~/Projects/project_template/conf/matplotlib/manuscript.mplstyle')
#%%
cell_annot = pd.read_csv("data/cell_type_align.txt", sep=',')
cell_annot['oc'] = cell_annot.tissue + '_' + cell_annot.celltype
cell_annot = cell_annot.set_index('id').to_dict()['oc']
motif_annot = pd.read_csv("data/motif_annot.tsv", sep='\t', names = ['cluter_id', 'cluster_name', 'motif', 'symbol', 'db', 'seq', 'Relative_orientation', 'Width', 'Left_offset', 'Right_offset'])
motif_clusters = pd.read_csv("data/motif_cluster.txt", sep='\t', names=['cluster'])['cluster'].values

plt.style.use('~/Projects/project_template/conf/matplotlib/manuscript.mplstyle')
# %%
fulco = pd.read_table("https://raw.githubusercontent.com/EngreitzLab/ABC-GWAS-Paper/main/comparePredictorsToCRISPRData/comparisonRuns/K562-only/experimentalData/experimentalData.K562-only.txt")
# fulco['distance'] = fulco['Gene TSS'] - (fulco.end + fulco.start)/2
# fulco['abc'] = fulco['H3K27ac (RPM)']/fulco['distance'].abs()
# %%
fulco = fulco.rename(columns={'chrPerturbationTarget':'Chromosome', 'startPerturbationTarget':'Start', 'endPerturbationTarget':'End', 'Gene':'Name'})
#%%
from pyliftover import LiftOver
lo = LiftOver('hg19', 'hg38')
for i, row in fulco.iterrows():
    fulco.loc[i, 'Start'] = lo.convert_coordinate(row['Chromosome'], row['Start'])[0][1]
    fulco.loc[i, 'End'] = lo.convert_coordinate(row['Chromosome'], row['End'])[0][1]
    fulco.loc[i, 'startTSS'] = lo.convert_coordinate(row['Chromosome'], row['startTSS'])[0][1]
    fulco.loc[i, 'endTSS'] = lo.convert_coordinate(row['Chromosome'], row['endTSS'])[0][1]

# %%
from pyranges import PyRanges as pr
from CellJacobian import CellJacobian
cell_jacob = CellJacobian('k562_count_10', 'Interpretation_k562_count10_hg38', True, True, 200, 283)
p = pr(fulco.reset_index()).join(pr(cell_jacob.peak_annot.reset_index()))

# %%
g = cell_jacob.gene_annot

g = g[g.gene_name.isin(fulco.GeneSymbol.unique())]

#%%
g = pd.merge(g, p.as_df(), left_on='gene_name', right_on='GeneSymbol')
#%%
g['id_dis'] = (g.level_0_x - g.level_0_y).abs()

#%%
g = g[g.id_dis<100]

#%%
# g[['gene_name', 'level_0_x', 'level_0_y', 'Strand', 'pred']].rename({'pred':'pred_original','level_0_x':'gene_idx', 'level_0_y':'re_idx', 'Strand':'strand'},axis=1).drop_duplicates().to_csv("Interpretation_mut/fulco_gene_re_idx.csv", index=False)
g['distance'] = g.endTSS - g.End
g= g[['gene_name', 'level_0_x', 'level_0_y', 'Strand', 'pred', 'EffectSize', 'Significant', 'distance']].rename({'pred':'pred_original','level_0_x':'gene_idx', 'level_0_y':'re_idx', 'Strand':'strand'},axis=1).drop_duplicates()
g.to_csv("Interpretation_mut/fulco_gene_re_idx.csv", index=False)
#%%
RESULT_DIR  = "Interpretation_mut/k562_count_10/"

obs = load_npz(os.path.join(RESULT_DIR, "obs.npz")).toarray().reshape(-1, 200, 2)

preds = load_npz(os.path.join(RESULT_DIR, "preds.npz")).toarray().reshape(-1, 200, 2)


g['obs'] = np.array([obs[i, 100, int(g.strand.iloc[i])] for i in range(len(g))])
g['pred_new'] = np.array([preds[i, 100, int(g.strand.iloc[i])] for i in range(len(g))])

g['delta'] = (g['pred_new'] - g['pred_original'])#.abs()#/g['distance']).abs()
g['log10fc_obs'] = g['EffectSize']
g['log10fc_pred'] = g['delta']
#%%
# g = g.groupby(['gene_name','re_idx', 'Significant']).mean().reset_index()

# g['delta'] = g['pred_new'] - g['pred_original']
#%%

sns.scatterplot(data=g[g.distance.abs()<300000], x='pred_new', y='pred_original', hue='Significant')
# add x=0 and y=0 line
# plt.plot([-1,1], [0,0], color='black', linestyle='--')
# plt.plot([0,0], [-1,1], color='black', linestyle='--')
# sns.scatterplot(data=g.groupby(['gene_name','re_idx'])[['Fraction change in gene expr','obs', 'Significant', 'pred_new', 'pred_original', 'delta']].mean(), x='delta', y='Fraction change in gene expr', hue='Significant', alpha=0.5, s=2)
#%%
# plot delta vs distance.abs() bined by 0,10000,500000,1000000,2000000
g['distance_bin'] = pd.cut(g.distance.abs(), bins=[0,1000,10000,50000,100000,500000,1000000,2000000], labels=['0-1k','1k-10k','10k-50k','50k-100k','100k-500k','500k-1M','1M-2M'])
sns.boxplot(data=g, x='distance_bin', y='delta', hue='Significant')
#%%
# facetgrid groupby distance_bin , scatterplot delta vs Fraction change in gene expr
g['distance_bin'] = pd.cut(g.distance.abs(), bins=[1000,10000,50000,100000,500000,1000000,2000000], labels=['1k-10k','10k-50k','50k-100k','100k-500k','500k-1M','1M-2M'])
sns.FacetGrid(data=g, col='distance_bin', col_wrap=2, height=2, sharex=False).map(sns.scatterplot, 'delta', 'log10fc_obs', alpha=0.5, s=10, hue=g.Significant)
#%%
# facetgrid groupby distance_bin , scatterplot delta vs Fraction change in gene expr
g['distance_bin'] = pd.cut(g.distance.abs(), bins=[1000,2000,4000,8000,16000,32000])
sns.FacetGrid(data=g, col='distance_bin', col_wrap=2, height=2, sharex=False).map(sns.scatterplot, 'delta', 'log10fc_obs', alpha=0.5, s=10, hue=g.Significant)
#%%
from scipy.stats import spearmanr
sns.scatterplot(x=g.delta,y=g.EffectSize, hue=g.Significant)

#%%
from sklearn.metrics import average_precision_score
average_precision_score(g[g.distance.abs()<5000]['Significant'], g[g.distance.abs()<5000]['delta'])
#%%
average_precision_score(g[g.distance.abs()<50000]['Significant'], g[g.distance.abs()<50000]['delta'])
#%%
average_precision_score(g[g.distance.abs()<100000]['Significant'], g[g.distance.abs()<100000]['delta'])
#%%
average_precision_score(g[g.distance.abs()<500000]['Significant'], g[g.distance.abs()<500000]['delta'])
#%%
average_precision_score(g[g.distance.abs()<5000]['Significant'], g[g.distance.abs()<5000]['abc'])
#%%
average_precision_score(g[g.distance.abs()<50000]['Significant'], g[g.distance.abs()<50000]['abc'])
#%%
average_precision_score(g[g.distance.abs()<100000]['Significant'], g[g.distance.abs()<100000]['abc'])
#%%
average_precision_score(g[g.distance.abs()<500000]['Significant'], g[g.distance.abs()<500000]['abc'])



#%%
# compute auroc based on 'Significant' column and 'ABC Score' column
from sklearn.metrics import roc_auc_score, average_precision_score
df = fulco#[fulco.dataset_na"me=='fulco2019']
roc_auc_score(df['Significant'], df['Fraction change in gene expr'].abs())
# %%
fulco[fulco.validated==1][['chromosome', 'enhancer_start', 'enhancer_end', 'gene', 'H3K27ac/abs(distance)']].to_csv("data/fulco_enhancers.bed", sep='\t', header=False, index=False)
# %%
#%%
focus=100

celltype = [
"k562",
]
#%%
from CellJacobian import CellJacobian
cell_jacobian_dict = {}
for celltype in tqdm(celltype):
    cell_jacobian_dict[celltype] = CellJacobian(celltype, jacob=True, num_region_per_sample=200, num_features=283)

# %%
from pyliftover import LiftOver
lo = LiftOver('hg19', 'hg38')
# %%
from pyranges import PyRanges as pr
# %%
def get_fulco_gene(gene):
    region_cell, preds, acces, regions = cell_jacobian_dict['k562'].query_gene(gene, keep='region')
    max_idx = int(np.argmax(preds, axis=0))
    for i in range(len(preds)):
        regions[i]['jacob'+str(i)] = region_cell.iloc[:,i].abs().values
    fulco_gene = fulco[fulco.Gene==gene][['chr', 'start', 'end', 'Gene', 'Significant', 'ABC Score', 'Fraction change in gene expr']].rename(columns={'chr':'Chromosome', 'start':'Start', 'end':'End', 'Gene':'Name'})
    fulco_gene['Start'] = fulco_gene['Start'].apply(lambda x: lo.convert_coordinate(fulco_gene.Chromosome.values[0], x)[0][1])
    fulco_gene['End'] = fulco_gene['End'].apply(lambda x: lo.convert_coordinate(fulco_gene.Chromosome.values[0], x)[0][1])
    print(fulco_gene)
    fulco_gene = pr(fulco_gene)
    print(fulco_gene)
    results = fulco_gene
    for i in range(len(regions)):
        results = results.join(pr(regions[i]))
    # results = pr(regions[max_idx]).join(fulco_gene).as_df()
    results = results.as_df()
    # results['jacob'] = results.loc[:,results.columns.str.contains('jacob')].mean(axis=1)
    return results
# %% 
gene= 'MYC'
df= get_fulco_gene(gene)
df['jacob'] = df.loc[:,df.columns.str.contains('jacob')].sum(axis=1)
gene_df = fulco[fulco.Gene==gene][['chr', 'start', 'end', 'Gene', 'Significant', 'ABC Score', 'Fraction change in gene expr']].rename(columns={'chr':'Chromosome', 'start':'Start', 'end':'End', 'Gene':'Name'})
# gene_df['Start'] = gene_df['Start'].apply(lambda x: lo.convert_coordinate(gene_df.Chromosome.values[0], x)[0][1])
# gene_df['End'] = gene_df['End'].apply(lambda x: lo.convert_coordinate(gene_df.Chromosome.values[0], x)[0][1])
gene_df = pr(gene_df)
df = gene_df.join(pr(df), how='left').as_df()
df.loc[df.jacob==-1, 'jacob']=0
#%%
fig, ax = plt.subplots(nrows=2, figsize=(5,5))
sns.scatterplot(x='Start', y= 'Fraction change in gene expr', data=df, hue = 'Significant', ax=ax[0])
sns.scatterplot(x='Start', y= 'jacob', data=df, hue = 'Significant', ax=ax[1])

#%%
sns.scatterplot(x='jacob', y= 'Fraction change in gene expr', data=df, hue = 'Significant')
# %%
df = df[df.relative_main_tss_distance.abs()<50000]
average_precision_score(df['validated'], df['ABC Score'])
# %%
fulco_result = []
aupr_jacobs = []
aupr_abcs = []
for gene in tqdm(fulco.Gene.unique()):
    try:
        df = get_fulco_gene(gene)
        df['jacob'] = df.loc[:,df.columns.str.contains('jacob')].sum(axis=1)
        fulco_result.append(df)
        # aupr_jacobs.append(aupr_jacob)
        # aupr_abcs.append(aupr_abc)
    except:
        pass
# %%
fulco_results = pd.concat([i[['Chromosome', 'Start', 'End', 'Name', 'Significant', 'ABC Score', 'Fraction change in gene expr', 'jacob']] for i in fulco_result], ignore_index=True)
# aupr_abcs = np.array(aupr_abcs)
# aupr_jacobs = np.array(aupr_jacobs)
#%%
sns.scatterplot(x='ABC Score', y= 'jacob', data=fulco_results, hue = 'Significant')
# %%
from sklearn.metrics import roc_auc_score
df = fulco_results[fulco_results.dataset_name!='fulco2019']
df = df.groupby(['Chromosome', 'Start', 'End', 'Name', 'validated', 'ABC Score', 'dataset_name', 'relative_main_tss_distance']).max().reset_index()
df = df[(df.relative_main_tss_distance.abs()<34500) & (df.relative_main_tss_distance.abs()<131000)]
average_precision_score(df['validated'], df['jacob'].abs())
#%%
average_precision_score(df['validated'], df['ABC Score'])
# %%
df.plot(x='ABC Score', y='jacob', kind='scatter')
# %%