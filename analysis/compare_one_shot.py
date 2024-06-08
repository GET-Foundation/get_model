# %%
tumor_sample_1 = "/home/ubuntu/alb2281/get/output/gbm_oneshot_gene_dea_Tumor.htan_gbm.C3N-01818_CPT0168270014_snATAC_GBM_Tumor.2048.csv"
tumor_sample_2 = "/home/ubuntu/alb2281/get/output/gbm_oneshot_gene_dea_Tumor.htan_gbm.C3N-02181_CPT0168380014_snATAC_GBM_Tumor.1024.csv"

# %%

import pandas as pd
names = ["gene", "value", "obs", "pred", "atpm"]
sample_1_df = pd.read_csv(tumor_sample_1, names=names)
sample_2_df = pd.read_csv(tumor_sample_2, names=names)
# %%
merged_df = pd.merge(sample_1_df, sample_2_df, on="gene", suffixes=("_1", "_2"))
# %%
merged_df["diff"] = merged_df["obs_1"] - merged_df["obs_2"]
# %%

merged_df.drop(columns=["value_1", "value_2"], inplace=True)
# %%
merged_df.set_index("gene")
# %%
merged_df = merged_df.sort_values(by="diff", ascending=False)
# %%
merged_df["abs_diff"] = merged_df["diff"].abs()
# %%
merged_df = merged_df.sort_values(by="abs_diff", ascending=False)
# %%

merged_df["preds_diff"] = merged_df["pred_1"] - merged_df["pred_2"]
# %%
merged_df["abs_preds_diff"] = merged_df["preds_diff"].abs()
# %%

import seaborn as sns
sns.scatterplot(data=merged_df, x="diff", y="preds_diff", s=1)

# %%
sns.scatterplot(data=merged_df, x="pred_1", y="preds_diff", s=1)
# %%
merged_df[['diff', 'preds_diff']].corr()
# %%
