#%%
import pandas as pd

result = pd.read_csv('/pmglocal/xf2217/output/k562_ref_fit_watac.csv', names=['gene_name', 'strand', 'tss_peak', 'perturb_chrom', 'perturb_start', 'perturb_end', 'pred_wt', 'pred_mut', 'obs'], header=None)
result
# %%
experiment = pd.read_csv('/burg/pmg/users/xf2217/CRISPR_comparison/resources/example/EPCrisprBenchmark_Fulco2019_K562_GRCh38.tsv.gz', sep='\t')
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
to_save
# %%
