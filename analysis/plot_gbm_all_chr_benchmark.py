# %%
import pandas as pd
import wandb
from tqdm import tqdm
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# %%

api = wandb.Api()
entity, project = "get-v3", "get-finetune-region-gbm-all-chr"
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

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

# filter to runs ending with chr1, chr2, chr3, chr4, chr5, ..., chr22, chrX 
runs_df = runs_df[runs_df["name"].str.contains("chr[0-9]")]



# %%
# drop runs beginning with DEBUG
runs_df = runs_df[~runs_df["name"].str.contains("DEBUG")]
# %%



runs_df["exp_pearson"] = runs_df["summary"].apply(lambda x: x["exp_pearson"] if "exp_pearson" in x else None)
runs_df["exp_spearman"] = runs_df["summary"].apply(lambda x: x["exp_spearman"] if "exp_spearman" in x else None)
runs_df["exp_r2"] = runs_df["summary"].apply(lambda x: x["exp_r2"] if "exp_r2" in x else None)
# %%
runs_df["exp"] = runs_df["name"].apply(lambda x: x.split("_")[-1])
# %%

runs_df = runs_df[["exp", "exp_pearson", "exp_spearman", "exp_r2"]]
# %%

runs_df.to_csv("/pmglocal/alb2281/repos/get_model/analysis/results/gbm_all_chr_benchmark.csv", index=False)
# %%

runs_df["exp_pearson"].describe()
# %%
