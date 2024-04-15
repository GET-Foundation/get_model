# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
metrics = pd.read_csv('~/output/val_metrics.csv')
# %%
pearson = metrics[['exp_pearson', 'count_filter', 'motif_scaler']].pivot(
    index='count_filter', columns='motif_scaler', values='exp_pearson')
# %%
# heatmap of pearson correlation
plt.figure(figsize=(5, 5))
# smaller font size for annot
sns.heatmap(pearson, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={
            'label': 'Pearson correlation'}, annot_kws={"size": 8})

# %%
spearman = metrics[['exp_spearman', 'count_filter', 'motif_scaler']].pivot(
    index='count_filter', columns='motif_scaler', values='exp_spearman')
# %%
# heatmap of pearson correlation
plt.figure(figsize=(5, 5))
# smaller font size for annot
sns.heatmap(spearman, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={
            'label': 'Spearman correlation'}, annot_kws={"size": 8})

# %%
r2 = metrics[['exp_r2', 'count_filter', 'motif_scaler']].pivot(
    index='count_filter', columns='motif_scaler', values='exp_r2')
# %%
# heatmap of pearson correlation
plt.figure(figsize=(5, 5))
# smaller font size for annot
sns.heatmap(r2, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={
            'label': 'R2'}, annot_kws={"size": 8})

# %%
