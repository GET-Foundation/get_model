# %%
import pandas as pd
import wandb
from tqdm import tqdm

# %%

api = wandb.Api()
entity, project = "get-v3", "get-finetune-region-gbm"
runs = api.runs(entity + "/" + project)

# %%

summary_list, config_list, name_list, metrics_list = [], [], [], []
for run in tqdm(runs):
    # .summary contains the output keys/values
    #  for metrics such as accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)
    metrics_list.append(run.history())
    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

# %%
runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list, "metrics": metrics_list}
)
# %%
finetune_df = runs_df[runs_df["name"].str.contains("finetune") & ~runs_df["name"].str.contains("DEBUG")]

# %%
finetune_df["chr"] = finetune_df["name"].apply(lambda x: x.split("_")[-1].split("chr")[1])
# %%
finetune_df = finetune_df[finetune_df["chr"] != "X"]

# %%

finetune_df["exp_pearson"] = finetune_df["metrics"].apply(lambda x: x["exp_pearson"].max())

# %%
finetune_df["exp_spearman"] = finetune_df["metrics"].apply(lambda x: x["exp_spearman"].max())

# %%
finetune_df["exp_r2"] = finetune_df["metrics"].apply(lambda x: x["exp_r2"].max())
# %%

chr_values = finetune_df["chr"].values.tolist()
exp_pearson_values = finetune_df["exp_pearson"].values.tolist()
exp_spearman_values = finetune_df["exp_spearman"].values.tolist()
exp_r2_values = finetune_df["exp_r2"].values.tolist()

# %%
import seaborn as sns
import matplotlib.pyplot as plt
# %%

import matplotlib.pyplot as plt
import numpy as np

# Sample data
groups = chr_values
variables = ['exp_pearson', 'exp_spearman', 'exp_r2']
data = np.random.rand(10, 3) * 100  # Random data for demonstration

# Setting up the figure and axis
fig, ax = plt.subplots()

# Width of each bar
bar_width = 0.25

# Position of bars on x-axis
index = np.arange(len(groups))

# Plotting bars for each variable
for i, var in enumerate([var1, var2, var3]):
    ax.bar(index + i * bar_width, var, bar_width, label=variables[i])

# Adding labels and title
ax.set_xlabel('Groups')
ax.set_ylabel('Values')
ax.set_title('Bar Plot with Three Variables and 10 Groups')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(groups)
ax.legend()

# Displaying the plot
plt.tight_layout()
plt.show()