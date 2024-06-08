# %%
import numpy as np 
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# %%
data_dir = "/home/ubuntu/alb2281/get/output/complete"
cols = ["gene", "value", "pred", "obs", "atpm"]

# %%
gbm_samples = [item for item in os.listdir(data_dir) if item.startswith("gbm")]

# %%

# read all of the csvs into a dataframe
dfs = []
for sample in gbm_samples:
    path = os.path.join(data_dir, sample)
    df = pd.read_csv(path, names=cols)
    df["sample"] = sample
    dfs.append(df)

# %%
all_dfs = pd.concat(dfs)
# %%
def process_sample_name(sample):
    parts = sample.split(".")
    method = parts[0].split("_")[1]
    celltype = parts[0].split("_")[-1]
    case_id = parts[2].split("_")[0]
    return f"{celltype}.{case_id}-{method}"
# %%
all_dfs["case_id"] = all_dfs["sample"].apply(process_sample_name)
# %%
all_dfs["method"] = all_dfs["case_id"].apply(lambda x: x.split("-")[-1])
# %%
# pivot the df so that the rows are genes and columns are case_id
pivoted_df = all_dfs.pivot(index="gene", columns="case_id", values="pred")
# %%
pivoted_df = pivoted_df.dropna()
# %%
# %%
fetal_path = "/home/ubuntu/alb2281/get/output/fetal_gene_expression_celltype.txt"
fetal_df = pd.read_csv(fetal_path, sep=",")
# %%

from caesar.io.gencode import Gencode
# %%
gencode = Gencode()
# %%
gencode.gtf
# %%
# get dictionary mapping gene_id to gene_name using gene_id and gene_name columns
gene_id_to_name = gencode.gtf[["gene_id", "gene_name"]].drop_duplicates().set_index("gene_id")["gene_name"].to_dict()
# %%
fetal_df.rename(columns={"RowID": "gene_id"}, inplace=True)
fetal_df["gene_id"] = fetal_df["gene_id"].apply(lambda x: x.split(".")[0])
fetal_df["gene_name"] = fetal_df["gene_id"].apply(lambda x: gene_id_to_name[x] if x in gene_id_to_name else None)
# %%

fetal_df.set_index("gene_name", inplace=True)

# %%
fetal_df.drop(columns=["gene_id"], inplace=True)
fetal_tpm = fetal_df.div(fetal_df.sum(axis=0), axis=1)
fetal_log_tpm = np.log10(1 + 1e6 * fetal_tpm)
# %%

fetal_log_tpm = fetal_log_tpm[["Cerebrum-Microglia", "Cerebrum-Astrocytes", "Cerebrum-Oligodendrocytes"]]
# %%

merged_df = pd.merge(pivoted_df, fetal_log_tpm, left_index=True, right_index=True, how="inner")

### ground truth

# %%
data_dir = "/home/ubuntu/alb2281/get/output/complete"
cols = ["gene", "value", "pred", "obs", "atpm"]

# %%
gbm_samples = [item for item in os.listdir(data_dir) if item.startswith("gbm")]

# %%

# read all of the csvs into a dataframe
dfs = []
for sample in gbm_samples:
    path = os.path.join(data_dir, sample)
    df = pd.read_csv(path, names=cols)
    df["sample"] = sample
    dfs.append(df)

# %%
all_dfs = pd.concat(dfs)
# %%
def process_sample_name(sample):
    parts = sample.split(".")
    method = parts[0].split("_")[1]
    celltype = parts[0].split("_")[-1]
    case_id = parts[2].split("_")[0]
    return f"{celltype}.{case_id}"
# %%
all_dfs["case_id"] = all_dfs["sample"].apply(process_sample_name)
# %%
# %%
all_dfs.drop(columns=["sample"], inplace=True)

# %%
all_dfs.drop(columns=["pred"], inplace=True)

# %%
all_dfs = all_dfs.drop_duplicates()

# %%
# pivot the df so that the rows are genes and columns are case_id
pivoted_gt_df = all_dfs.pivot(index="gene", columns="case_id", values="obs")
# %%
pivoted_gt_df = pivoted_gt_df.dropna()
# %%

all_df = pd.merge(pivoted_gt_df, merged_df, left_index=True, right_index=True, how="inner")
# %%

# filter to columns with Tumor in name
tumor_cols = [col for col in all_df.columns if "Tumor" in col]
tumor_df = all_df[tumor_cols]

# %%

# separate into two df for paired t test between ground truth and oneshot
gt = tumor_df[[col for col in tumor_df.columns if "oneshot" not in col and "zeroshot" not in col]]
oneshot = tumor_df[[col for col in tumor_df.columns if "oneshot" in col]]
# rename oneshot columns to remove "oneshot"
oneshot.columns = [col.split("-")[0] + "-" + col.split("-")[1] for col in oneshot.columns]



# %%
# sort the columns
gt = gt[sorted(gt.columns)]
oneshot = oneshot[sorted(oneshot.columns)]

gt = gt.drop_duplicates()
oneshot = oneshot.drop_duplicates()


# %%
from scipy.stats import ttest_rel
pvals = []
pvals_index = []
for gene in tqdm(gt.index):
    result = ttest_rel(gt.loc[gene], oneshot.loc[gene]).pvalue
    # convert result to float
    pvals.append(float(result))
    pvals_index.append(gene)

# %%
# read in pvals and pvals_index as columns of a dataframe
pval_df = pd.DataFrame({"gene": pvals_index, "pval": pvals})

# %%
# bh adjust
from statsmodels.stats.multitest import multipletests
pval_df['bh'] = multipletests(pval_df['pval'], method='fdr_bh')[1]

# %%

sig_genes = pval_df[pval_df["bh"]<0.0001]["gene"]

# %%

df_to_plot = tumor_df[tumor_df.quantile(0.25,axis=1)>1].query("index in @sig_genes")
sns.clustermap(df_to_plot, cmap="viridis", yticklabels=True, xticklabels=True, metric="correlation")
# %%

tumor_df_without_zeroshot = tumor_df[[col for col in tumor_df.columns if "zeroshot" not in col]]
# %%
tumor_df_without_zeroshot = tumor_df_without_zeroshot.drop_duplicates()
# %%
df_to_plot = tumor_df.query("index in @sig_genes")
sns.clustermap(tumor_df_without_zeroshot, cmap="viridis", yticklabels=True, xticklabels=True, metric="correlation")
# %%
df_to_plot = tumor_df[tumor_df.quantile(0.25,axis=1)>1].query("index in @sig_genes")


# %%
# remove "zeroshot" columns
df_to_plot = df_to_plot[[col for col in df_to_plot.columns if "zeroshot" not in col]]
sns.clustermap(df_to_plot, cmap="viridis", yticklabels=True, xticklabels=True, metric="correlation", col_cluster="False")
# %%

# sort columns by name
df_to_plot = df_to_plot[sorted(df_to_plot.columns)]
sns.clustermap(df_to_plot, cmap="viridis", yticklabels=True, xticklabels=True, metric="correlation", col_cluster=False)
# %%
