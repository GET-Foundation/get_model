# %%
import numpy as np 
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import fcluster

# %%
zeroshot_data_dir = "/burg/pmg/users/alb2281/get/results/gbm_deg_v2_lora/watac-zeroshot-astrocyte"
oneshot_data_dir = "/burg/pmg/users/alb2281/get/results/gbm_deg_v2_lora/watac-oneshot-gbm"
cols = ["gene", "value", "pred", "obs", "atpm"]

# %%
zeroshot_gbm_samples = [item for item in os.listdir(zeroshot_data_dir) if item.startswith("gbm")]
oneshot_gbm_samples = [item for item in os.listdir(oneshot_data_dir) if item.startswith("gbm")]
# %%

# read all of the csvs into a dataframe
zeroshot_dfs = []
for sample in zeroshot_gbm_samples:
    path = os.path.join(zeroshot_data_dir, sample)
    df = pd.read_csv(path, names=cols)
    df["sample"] = sample
    zeroshot_dfs.append(df)

oneshot_dfs = []
for sample in oneshot_gbm_samples:
    path = os.path.join(oneshot_data_dir, sample)
    df = pd.read_csv(path, names=cols)
    df["sample"] = sample
    oneshot_dfs.append(df)


# %%
dfs = zeroshot_dfs + oneshot_dfs
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
# filter to only oneshot and samples with Tumor in the case_id
all_dfs = all_dfs[all_dfs["method"] == "oneshot"]
all_dfs = all_dfs[all_dfs["case_id"].str.contains("Tumor")]

# %%
# pivot the df so that the rows are genes and columns are case_id
pivoted_df = all_dfs.pivot(index="gene", columns="case_id", values="obs")
# %%
pivoted_df = pivoted_df.dropna()
# %%
# get list of most variable rows
most_variable = pivoted_df.var(axis=1).sort_values(ascending=False).index[:50]
# %%
subset_df = pivoted_df.loc[most_variable]
# %%

g = sns.clustermap(subset_df, cmap="viridis", method="ward", yticklabels=True, xticklabels=True, figsize=(20, 20))

# %%
linkage_matrix = g.dendrogram_col.linkage
# %%

# filter subset_df to only tumor
tumor_df = subset_df[[col for col in subset_df.columns if "Tumor" in col]]
# %%
# Define the number of clusters or the distance threshold
num_clusters = 3 # You can change this as needed

# Retrieve cluster IDs
cluster_ids = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
# %%

tumor_to_cluster = dict(zip(tumor_df.columns, cluster_ids))
# %%
