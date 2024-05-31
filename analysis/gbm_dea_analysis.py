# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %%

fetal_file = "/home/ubuntu/alb2281/get/output/fetal_gene_expression_celltype.txt"
zeroshot_file = "/home/ubuntu/alb2281/get/output/gbm_zeroshot_gene_dea_Tumor.htan_gbm.C3N-01818_CPT0168270014_snATAC_GBM_Tumor.2048.csv"
oneshot_file = "/home/ubuntu/alb2281/get/output/gbm_oneshot_gene_dea_Tumor.htan_gbm.C3N-01818_CPT0168270014_snATAC_GBM_Tumor.2048.csv"

# %%
fetal_df = pd.read_csv(fetal_file, sep=",")
# %%

fetal_df
# %%
from caesar.io.gencode import Gencode
# %%
gencode = Gencode(gtf_dir="/home/ubuntu/alb2281/get/get_data")

# %%
gencode.gtf
# %%
# get a dictionary from gene_id to gene_name from gencode.gtf dataframe
gene_id_to_name = gencode.gtf.set_index("gene_id")["gene_name"].to_dict()
# %%
# map gene_id to gene_name else None if not in the dict
fetal_df["gene_id"] = fetal_df["RowID"].apply(lambda x: x.split(".")[0])
fetal_df["gene_name"] = fetal_df["gene_id"].apply(lambda x: gene_id_to_name.get(x, None))
# %%
fetal_astrocyte_df = fetal_df[["gene_name", "Cerebellum-Astrocytes"]]
# %%
fetal_astrocyte_df["Cerebellum-Astrocytes-TPM"] = fetal_astrocyte_df["Cerebellum-Astrocytes"].apply(lambda x: np.log10(1 + 1e6 * x/np.sum(fetal_astrocyte_df["Cerebellum-Astrocytes"])))
# %%
fetal_astrocyte_df["Cerebellum-Astrocytes-TPM"].max()
# %%
fetal_astrocyte_df["Cerebellum-Astrocytes-TPM"].min()
# %%
columns = ["gene_name", "value", "pred", "obs", "atpm"]
zeroshot_df = pd.read_csv(zeroshot_file, sep=",", names=columns)
# %%
oneshot_df = pd.read_csv(oneshot_file, sep=",", names=columns)
# %%
merged_df = fetal_astrocyte_df.merge(zeroshot_df, on="gene_name", how="inner", suffixes=("_astrocyte", "_zeroshot"))
# %%
zeroshot_df.drop(columns=["value"])
# %%
merged_df = fetal_astrocyte_df.merge(zeroshot_df, on="gene_name", how="inner")
# %%
merged_df = merged_df.merge(oneshot_df, on="gene_name", how="inner", suffixes=("_zeroshot", "_oneshot"))
# %%

merged_df = merged_df.drop(columns=["obs_oneshot", "atpm_oneshot"])
# %%
merged_df = merged_df.rename(columns={"obs_zeroshot": "exp_obs", "atpm_zeroshot": "atpm"})
# %%

merged_df = merged_df[["gene_name", "Cerebellum-Astrocytes-TPM", "exp_obs", "pred_zeroshot", "pred_oneshot", "atpm"]]
# %%
merged_df = merged_df.rename(columns={"exp_obs": "gbm_obs"})
# %%
import seaborn as sns
# %%
sns.scatterplot(data=merged_df, x="Cerebellum-Astrocytes-TPM", y="gbm_obs", s=1)
# %%
sns.scatterplot(data=merged_df, x="gbm_obs", y="pred_zeroshot", s=1)
# %%
sns.scatterplot(data=merged_df, x="gbm_obs", y="pred_oneshot", s=1)
# %%
# compute pearson correlation between fetal astrocyte TPM and gbm obs
from scipy.stats import pearsonr
pearsonr(merged_df["Cerebellum-Astrocytes-TPM"], merged_df["gbm_obs"])

# %%
pearsonr(merged_df["pred_zeroshot"], merged_df["gbm_obs"])
# %%
pearsonr(merged_df["pred_oneshot"], merged_df["gbm_obs"])
# %%
merged_df["pred_diff"] = merged_df["pred_oneshot"] - merged_df["pred_zeroshot"]
# %%
merged_df = merged_df.sort_values(by="pred_diff", ascending=False)
# %%
merged_df
# %%
# query gene equal to EGFR
merged_df.query("gene_name == 'EGFR'")
# %%

# show pearson correlation between fetal astrocyte TPM and gbm obs on scatterplot

def scatterplot_with_pearson(data, x, y, s=1, title=None):
    sns.scatterplot(data=data, x=x, y=y, s=s)
    corr = pearsonr(data[x], data[y])
    plt.title("Pearson r: " + str(corr.statistic))
    plt.show()
# %%

scatterplot_with_pearson(merged_df, "Cerebellum-Astrocytes-TPM", "gbm_obs", s=1)
scatterplot_with_pearson(merged_df, "gbm_obs", "pred_zeroshot", s=1)
scatterplot_with_pearson(merged_df, "gbm_obs", "pred_oneshot", s=1)
# %%
