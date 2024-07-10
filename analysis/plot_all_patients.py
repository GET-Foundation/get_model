# %%
import pandas as pd
import wandb
from tqdm import tqdm
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# %%

api = wandb.Api()
entity, project = "get-v3", "get-zeroshot-gbm-all-celltypes"
runs = api.runs(entity + "/" + project)

# %%
summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary contains the output keys/values
    #  for metrics such as accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

zeroshot_runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)
# %%
zeroshot_runs_df["exp_pearson"] = zeroshot_runs_df["summary"].apply(lambda x: x["exp_pearson"] if "exp_pearson" in x else None)
zeroshot_runs_df["exp_spearman"] = zeroshot_runs_df["summary"].apply(lambda x: x["exp_spearman"] if "exp_spearman" in x else None)
zeroshot_runs_df["exp_r2"] = zeroshot_runs_df["summary"].apply(lambda x: x["exp_r2"] if "exp_r2" in x else None)
# %%
zeroshot_runs_df["case_id"] = zeroshot_runs_df["name"].apply(lambda x: x.split(".")[-2].split("_")[0])


# %%
zeroshot_runs_df = zeroshot_runs_df.dropna()
# %%

api = wandb.Api()
entity, project = "get-v3", "get-zeroshot-gbm-finetuned-ckpt"
runs = api.runs(entity + "/" + project)

# %%
summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary contains the output keys/values
    #  for metrics such as accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

oneshot_runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)
# %%
oneshot_runs_df["exp_pearson"] = oneshot_runs_df["summary"].apply(lambda x: x["exp_pearson"] if "exp_pearson" in x else None)
oneshot_runs_df["exp_spearman"] = oneshot_runs_df["summary"].apply(lambda x: x["exp_spearman"] if "exp_spearman" in x else None)
oneshot_runs_df["exp_r2"] = oneshot_runs_df["summary"].apply(lambda x: x["exp_r2"] if "exp_r2" in x else None)
# %%
oneshot_runs_df["case_id"] = oneshot_runs_df["name"].apply(lambda x: x.split(".")[-2].split("_")[0])


# %%
oneshot_runs_df = oneshot_runs_df.dropna()


# %%
def get_case_id(x):
    parts = x.split(".")
    celltype = parts[-4].split("_")[-1]
    case_id = parts[-2].split("_")[0]
    return f"{celltype}.{case_id}"
# %%
zeroshot_runs_df["case_id"] = zeroshot_runs_df["name"].apply(get_case_id)
# %%

oneshot_runs_df["case_id"] = oneshot_runs_df["name"].apply(get_case_id)
# %%

merged_df = pd.merge(oneshot_runs_df, zeroshot_runs_df, on="case_id", suffixes=("_oneshot", "_zeroshot"))
# %%

merged_df = merged_df.dropna()
# %%
merged_df["celltype"] = merged_df["case_id"].apply(lambda x: x.split(".")[0])
# %%
merged_df["patient_id"] = merged_df["case_id"].apply(lambda x: x.split(".")[1])


# %%
merged_df = merged_df[['case_id', 'celltype', "patient_id", 'exp_pearson_zeroshot', 'exp_pearson_oneshot']]

# %%
# take the merged_df and unfold it into three columns celltype, zeroshot or oneshot, and the value
unfolded_df = pd.melt(merged_df, id_vars=["celltype", "patient_id", "case_id"], value_vars=["exp_pearson_oneshot", "exp_pearson_zeroshot"], var_name="method", value_name="exp_pearson")
# %%
plt.figure(figsize=(10, 6))
sns.violinplot(x='celltype', y='exp_pearson', hue='method', data=unfolded_df, split=True)
plt.title('Violin Plot of Different Cell Types by Treatment')
plt.show()

# %%
unfolded_df = unfolded_df[unfolded_df["patient_id"] != "C3L-03405"]
# %%

# put all rows with celltype Tumor first in unfolded_df

unfolded_df["method"] = unfolded_df["method"].apply(lambda x: x.split("_")[2])

# %%
method_order = ["zeroshot", "oneshot"]
celltype_order = ["Tumor", "Macrophages", "Neurons", "Oligodendrocytes"]
# %%
df_sorted = unfolded_df.sort_values(
    by=['method', 'celltype'],
    key=lambda x: x.map({**{v: i for i, v in enumerate(method_order)}, **{v: i for i, v in enumerate(celltype_order)}})
)

# %%
plt.figure(figsize=(10, 6))
sns.violinplot(x='celltype', y='exp_pearson', hue='method', data=df_sorted, split=True, inner="stick")
plt.xlabel("Cell Type")
plt.ylabel("Pearson correlation")
plt.legend(title="")
plt.title('Violin Plot of Different Cell Types by Treatment')
plt.show()
# %%

method_name_map = {
    "zeroshot": "Zero-shot",
    "oneshot": "One-shot (finetuned)"
}
# %%
df_sorted = df_sorted.replace({"method": method_name_map})
# %%
plt.figure(figsize=(10, 6))
sns.violinplot(x='celltype', y='exp_pearson', hue='method', data=df_sorted, split=True, inner="stick")
plt.xlabel("Cell Type")
plt.ylabel("Pearson correlation")
plt.legend(title="")
plt.title('Zero-shot vs. one-shot performance on leave-out GBM samples')
plt.show()
# %%
