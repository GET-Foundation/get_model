# %%
import numpy as np 
import pandas as pd
import seaborn as sns

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
# %%
tumor_cerebrum_df = merged_df[[col for col in merged_df.columns if "Tumor" in col or "Cerebrum" in col]]

# %%
# get list of most variable rows
most_variable = tumor_cerebrum_df.var(axis=1).sort_values(ascending=False).index[:200]
# %%
subset_df = tumor_cerebrum_df.loc[most_variable]

# %%
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
pivoted_df = all_dfs.pivot(index="gene", columns="case_id", values="obs")
# %%
pivoted_df = pivoted_df.dropna()
# %%
# get list of most variable rows

# merge with merged_df
merged_df = pd.merge(pivoted_df, subset_df, left_index=True, right_index=True, how="inner")
# %%
merged_df.iloc["EGFR"]



# %%
tumor_cerebrum_df = merged_df[[col for col in merged_df.columns if "Tumor" in col or "Cerebrum" in col]]
# %%
tumor_cerebrum_df.to_csv("/home/ubuntu/alb2281/analysis/output_data/gbm_most_variable_genes_all_samples_preds_with_gt_tumor_only.csv")
# %%
