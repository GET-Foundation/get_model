# %%
import pandas as pd 
import seaborn as sns

# %%
names = ["gene", "val", "pred", "obs", "atpm"]

# %%
train_sample = "/pmglocal/alb2281/get/output/watac-oneshot-gbm/gbm_oneshot_gene_dea_Tumor.htan_gbm.C3L-03405_CPT0224600013_snATAC_GBM_Tumor.4096_watac.csv"
sample_1_oneshot = "/pmglocal/alb2281/get/output/watac-oneshot-gbm/gbm_oneshot_gene_dea_Tumor.htan_gbm.C3L-03968_CPT0228220004_snATAC_GBM_Tumor.4096_watac.csv"
sample_2_oneshot = "/pmglocal/alb2281/get/output/watac-oneshot-gbm/gbm_oneshot_gene_dea_Tumor.htan_gbm.C3N-01814_CPT0167860015_snATAC_GBM_Tumor.16384_watac.csv"

# %%
train_sample = pd.read_csv(train_sample, names=names)
sample_1_oneshot = pd.read_csv(sample_1_oneshot, names=names)
sample_2_oneshot = pd.read_csv(sample_2_oneshot, names=names)
# %%
merged_df = sample_1_oneshot.merge(sample_2_oneshot, on="gene", suffixes=("_sample1", "_sample2"))
# %%
merged_df["log2fc"] = merged_df["pred_sample1"] - merged_df["pred_sample2"]
merged_df["log2fc_obs"] = merged_df["obs_sample1"] - merged_df["obs_sample2"]
merged_df['log2fc_atpm'] = merged_df['atpm_sample1'] - merged_df['atpm_sample2']
#%%
sns.scatterplot(data=merged_df.query("log2fc.abs()>1"), x="log2fc_obs", y="log2fc", hue='log2fc_atpm', s=5)
# pearson between log2fc and log2fc_obs
merged_df[["log2fc_obs", "log2fc", "log2fc_atpm"]].corr()
# %%
merged_df.query("log2fc.abs()>1 & (pred_sample1>1 | pred_sample2>1) ").sort_values("log2fc", ascending=False)

# %%
oneshot_output = "/pmglocal/alb2281/get/output/watac-oneshot-gbm/gbm_oneshot_gene_dea_Tumor.htan_gbm.C3L-03405_CPT0224600013_snATAC_GBM_Tumor.4096_watac.csv"
# %%
oneshot_output = pd.read_csv(oneshot_output, names=names)
sns.scatterplot(data=oneshot_output, x="obs", y="pred", s=1)
# %%
