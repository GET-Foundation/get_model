#%%
import pandas as pd
#%%
result = pd.read_csv('/pmglocal/xf2217/output/k562_ref_fit_watac.csv', names=['gene_name', 'strand', 'tss_peak', 'perturb_chrom', 'perturb_start', 'perturb_end', 'pred_wt', 'pred_mut', 'obs'], header=None)
result
# %%
experiment = pd.read_csv('../EPCrisprBenchmark_Fulco2019_K562_GRCh38.tsv.gz', sep='\t')
# %%
experiment['key'] = experiment['chrom'] + ':' + experiment['chromStart'].astype(str) + '-' + experiment['chromEnd'].astype(str) + ':' + experiment['measuredGeneSymbol']

# %%
result['key'] = result['perturb_chrom'] + ':' + result['perturb_start'].astype(str) + '-' + result['perturb_end'].astype(str) + ':' + result['gene_name']
# %%
import numpy as np
merged = pd.merge(result, experiment, on='key', how='left')
merged['pred_mut'] = merged['pred_mut'].astype(float)
merged['pred_wt'] = merged['pred_wt'].astype(float)
merged['obs'] = merged['obs'].astype(float)
merged['Distance'] = (merged['startTSS'] - merged['perturb_start']).abs()
merged['logfc'] = (merged['pred_mut'] - merged['pred_wt']).abs()/merged['pred_wt']#/merged['Distance']
# merged = merged.query('startTSS-perturb_start>100_000')
# merged = merged.query('pred_wt>0.1').dropna()
# precision recall curve of logfc to significant
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
precision, recall, _ = precision_recall_curve(merged['Significant'], merged['logfc'])
average_precision = average_precision_score(merged['Significant'], merged['logfc'])
plt.plot(recall, precision, label='AP={0:0.3f}'.format(average_precision))
# add random
plt.plot([0, 1], [merged['Significant'].mean(), merged['Significant'].mean()], linestyle='--', label='Random={0:0.3f}'.format(merged['Significant'].mean()))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend()
plt.show()

# %%
import seaborn as sns
sns.scatterplot(x='logfc', y='EffectSize', data=merged, hue='Significant')
# %%
merged[['logfc', 'EffectSize']].corr(method='spearman')
# %%
# group by gene_name and calculate the correlation
correlation = merged.groupby('gene_name')[['logfc', 'EffectSize']].corr(method='spearman').reset_index().query('level_1=="logfc"').dropna()
# %%
correlation
# %%
acc = merged[['gene_name', 'obs','pred_wt']].drop_duplicates()
acc['error'] = (acc['obs'] - acc['pred_wt'])/acc['obs']
# %%
pd.merge(correlation, acc, on='gene_name').plot(x='error', y='EffectSize', kind='scatter')
# %%
merged.rename({'chromStart':'start', 'chromEnd':'end', 'chrom':'chr', 'TargetGene': 'gene_name', 'logfc':'GET'}, axis=1)
# %%
to_save = merged.rename({'chromStart':'start', 'chromEnd':'end', 'chrom':'chr', 'measuredGeneSymbol': 'TargetGene', 'logfc':'GET'}, axis=1)[['chr', 'start', 'end', 'TargetGene', 'GET']]
to_save['CellType'] = 'K562'
to_save.to_csv('/burg/pmg/users/xf2217/CRISPR_comparison/resources/example/GETQATAC_K562_Fulco2019Genes_GRCh38.tsv', sep='\t')
# %%
to_save
# %%
# load jacobian-based analysis
import zarr
z = zarr.open_group('/home/xf2217/output/k562_lora_captum.zarr/', mode='r')
#%%
import numpy as np
gene_names = [x.strip(' ') for x in z['gene_name'][:]]
chromosomes = [x.strip(' ') for x in z['chromosome'][:]]
peaks = z['peaks'][:]
strand = z['strand'][:]
#%%
jacobians = np.stack([z['jacobians']['exp'][str(x)]['input'][i] for i, x in enumerate(strand)])
jacobians = jacobians - jacobians.mean(0)
jacobians = jacobians / jacobians.std(0)
jacobians[jacobians<0] = 0
input_data = z['input'][:]
atac = input_data[:,:, 282]
peaks = z['peaks'][:]
input_x_jacob = jacobians

# %%
dfs = []
import pandas as pd
for i, gene in enumerate(gene_names):
    gene_df = pd.DataFrame({'score': input_x_jacob[i].sum(1), 'Chromosome': chromosomes[i], 'Start': peaks[i][:, 0], 'End': peaks[i][:, 1], 'gene_name': gene, 'Strand': strand[i], 'atac': atac[i]})
    dfs.append(gene_df)


# %%
dfs = pd.concat(dfs).query('Start>0')
# %%
dfs
# %%


#%%
import zarr
z = zarr.open_group('/home/xf2217/output/k562_lora_captum_natac.zarr/', mode='r')
import numpy as np
gene_names = [x.strip(' ') for x in z['gene_name'][:]]
chromosomes = [x.strip(' ') for x in z['chromosome'][:]]
peaks = z['peaks'][:]
strand = z['strand'][:]
attribution = z['attribution'][:].reshape(-1,900)
input = z['input'][:]
atac = input[:,:,282]
dfs = []
import pandas as pd
for i, gene in enumerate(gene_names):
    gene_df = pd.DataFrame({'score': attribution[i] , 'Chromosome': chromosomes[i], 'Start': peaks[i][:, 0], 'End': peaks[i][:, 1], 'gene_name': gene, 'Strand': strand[i], 'atac': atac[i]})
    dfs.append(gene_df)

dfs = pd.concat(dfs).query('Start>0')
dfs = dfs.groupby(['Chromosome', 'Start', 'End', 'gene_name', 'Strand']).mean().reset_index()
# normalize score per gene
dfs['score'] = (dfs['score'] - dfs.groupby('gene_name')['score'].transform('min')) / (dfs.groupby('gene_name')['score'].transform('max') - dfs.groupby('gene_name')['score'].transform('min'))
#%%
experiment = experiment.rename({'chrom':'Chromosome', 'chromStart':'Start', 'chromEnd':'End', 'measuredGeneSymbol':'gene_name'}, axis=1)
# %%
experiment
# %%
from pyranges import PyRanges as pr

overlap = pr(experiment).join(pr(dfs), suffix='_experiment').df.query('gene_name_experiment==gene_name')
overlap['Distance'] = (overlap['Start'] - overlap['startTSS']).abs()
#%%
overlap['logdis'] = np.log10(overlap['Distance'])
overlap['logp'] = -np.log10(overlap['pValueAdjusted'])
#%%
import seaborn as sns
sns.scatterplot(x='atac', y='score', data=overlap, hue='Reference')
#%%
hic_gamma = 1.024238616787792
hic_scale = 5.9594510043736655
hic_gamma_reference = 0.87
hic_scale_reference = -4.80 + 11.63 * hic_gamma_reference

def get_powerlaw_at_distance(distances, gamma, scale, min_distance=5000):
    assert gamma > 0
    assert scale > 0

    # The powerlaw is computed for distances > 5kb. We don't know what the contact freq looks like at < 5kb.
    # So just assume that everything at < 5kb is equal to 5kb.
    # TO DO: get more accurate powerlaw at < 5kb
    distances = np.clip(distances, min_distance, np.Inf)
    log_dists = np.log(distances + 1)

    powerlaw_contact = np.exp(scale + -1 * gamma * log_dists)
    return powerlaw_contact

overlap['score'] = overlap['score'].fillna(0) * overlap['atac']
overlap['powerlaw'] = get_powerlaw_at_distance(overlap['Distance'], hic_gamma_reference, hic_scale_reference)
overlap['atac/dis'] = overlap['atac'] * (np.exp(-overlap['Distance']/1000))
overlap['score*atac'] = overlap['atac'] * overlap['score']
overlap['score/dis'] = overlap['score'] * (np.exp(-overlap['Distance']/1000))
# distance with cauchy distribution

# overlap['atac/dis'] = overlap['atac'] * (np.exp(-overlap['Distance']**2/(2*1000**2))/ np.sqrt(2*np.pi * 1000**2)) 
# overlap['score/dis'] = overlap['score'] * (np.exp(-overlap['Distance']**2/(2*1000**2))/ np.sqrt(2*np.pi * 1000**2))

overlap['atac/dis'] = overlap['atac'] / overlap['Distance']
overlap['score/dis'] = overlap['score'] / overlap['Distance']
# sns.scatterplot(x='Distance', y='score', data=overlap)
# exponential distribution
# %%
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
# overlap=overlap.query('Distance>10_000')
# overlap = overlap.query('Reference=="Fulco2019"')
precision, recall, _ = precision_recall_curve(overlap['Significant'], overlap['score/dis'])
average_precision = average_precision_score(overlap['Significant'], overlap['score/dis'])
atac_precision, atac_recall, _ = precision_recall_curve(overlap['Significant'], overlap['atac/dis'])
atac_average_precision = average_precision_score(overlap['Significant'], overlap['atac/dis'])
atac_only_precision, atac_only_recall, _ = precision_recall_curve(overlap['Significant'], overlap['atac'])
atac_only_average_precision = average_precision_score(overlap['Significant'], overlap['atac'])

score_only_precision, score_only_recall, _ = precision_recall_curve(overlap['Significant'], overlap['score'])
score_only_average_precision = average_precision_score(overlap['Significant'], overlap['score'])
# distance_precision, distance_recall, _ = precision_recall_curve(overlap['Significant'], 1/overlap['Distance'])
# distance_average_precision = average_precision_score(overlap['Significant'], 1/overlap['Distance'])
plt.plot(recall, precision, label='GET/Dis AP={0:0.3f}'.format(average_precision))
plt.plot(atac_only_recall, atac_only_precision, label='ATAC AP={0:0.3f}'.format(atac_only_average_precision), linestyle='--')
plt.plot(score_only_recall, score_only_precision, label='GET AP={0:0.3f}'.format(score_only_average_precision), linestyle='--')
plt.plot(atac_recall, atac_precision, label='ATAC/Dis AP={0:0.3f}'.format(atac_average_precision), linestyle='--')

# add random
plt.plot([0, 1], [overlap['Significant'].mean(), overlap['Significant'].mean()], linestyle='--', label='Random={0:0.3f}'.format(overlap['Significant'].mean()))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend()
plt.show()
# %%
to_save = overlap.rename({'Start_experiment':'start', 'End_experiment':'end', 'Chromosome':'chr', 'gene_name': 'TargetGene', 'atac/dis':'AtacDis'}, axis=1)[['chr', 'start', 'end', 'TargetGene', 'AtacDis']]
to_save['CellType'] = 'K562'
to_save.to_csv('/burg/pmg/users/xf2217/CRISPR_comparison/resources/example/AtacDis_K562_Fulco2019Genes_GRCh38.tsv', sep='\t')
# %%
sns.scatterplot(x=input_data[np.random.choice(1800, 1000)].mean(0).flatten(), y =input_data[np.random.choice(1800, 1000)].mean(0).flatten())
# %%
np.save('random_input_for_k562.npz', input_data[np.random.choice(1800, 1000)].mean(0))
# %%
import numpy as np
np.load('random_input_for_k562.npy')
# %%
overlap.plot(x='score',y='atac',kind='scatter')
# %%
