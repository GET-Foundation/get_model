# %%
data_dir = "/home/ubuntu/alb2281/get/output"
# %%
import os
import pandas as pd 

# %%
gbm_samples = [item for item in os.listdir(data_dir) if item.startswith("gbm_oneshot")]
# %%
# read all of the csvs into a dataframe
dfs = []
cols=["gene", "value", "pred", "obs", "atpm"]
for sample in gbm_samples:
    path = os.path.join(data_dir, sample)
    df = pd.read_csv(path, names=cols)
    df["sample"] = sample
    dfs.append(df)

# %%
all_dfs = pd.concat(dfs)
# %%
# pivot so that the rows are genes and the columns are samples
all_dfs_pred = all_dfs.pivot(index="gene", columns="sample", values="pred")
all_dfs_obs = all_dfs.pivot(index="gene", columns="sample", values="obs")
all_dfs_atpm = all_dfs.pivot(index="gene", columns="sample", values="atpm")
# %%

# %%
def process_sample_name(x):
    parts = x.split(".")   
    celltype = parts[0].split("_")[-1]
    case_id = parts[2].split("_")[0]
    return f"{celltype}.{case_id}"


# apply process_sample_name to the column names
all_dfs_pred.columns = all_dfs_pred.columns.map(lambda x: process_sample_name(x))
all_dfs_obs.columns = all_dfs_obs.columns.map(lambda x: process_sample_name(x))
all_dfs_atpm.columns = all_dfs_atpm.columns.map(lambda x: process_sample_name(x))

# pivot so that the rows are genes and the columns are samples

# %%
merged_df = pd.merge(all_dfs_pred, all_dfs_obs, left_index=True, right_index=True, suffixes=("_pred", "_obs"))
# %%


sample_list = list(all_dfs_pred.columns)
pred_diffs = []
obs_diffs = []
for i in range(len(sample_list)):
    for j in range(i+1, len(merged_df.columns)):
        sample1 = merged_df.columns[i]
        sample2 = merged_df.columns[j]
        print(sample1)
        print(sample2)
        pred_diffs.append(merged_df[f"{sample1}_pred"] - merged_df[f"{sample2}_pred"])
        obs_diffs.append(merged_df[f"{sample1}_obs"] - merged_df[f"{sample2}_obs"])
        

# %%
