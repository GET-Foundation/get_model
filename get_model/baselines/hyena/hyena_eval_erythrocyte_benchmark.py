# %%
import json 
import pandas as pd
import numpy as np

# %%
results_dir = "/burg/pmg/users/alb2281/hyena/results/erythroblast"


all_preds = []

for file in os.listdir(results_dir):
    with open(os.path.join(results_dir, file), "r") as f:
        data = json.load(f)
        for d in data:
            all_preds.append(d)

# %%

# read list of dicts into dataframe
hyena_df = pd.DataFrame(all_preds)
# %%

hyena_df = hyena_df.drop_duplicates(subset="orig_idx", keep="last")
# %%

hyena_df = hyena_df.sort_values(by="orig_idx")

# %%
def compute_mean_score(row):
    if len(row["region_logits"]) == 0:
        return np.nan
    else:
        return np.mean(row["knockout_logits"]) - np.mean(row["region_logits"])
    
def compute_sum_score(row):
    if len(row["region_logits"]) == 0:
        return np.nan
    else:
        return np.sum(row["knockout_logits"]) - np.sum(row["region_logits"])
# %%
merged_df = hyena_df.copy()
merged_df["mean_score"] = merged_df.apply(compute_mean_score, axis=1)
merged_df["sum_score"] = merged_df.apply(compute_sum_score, axis=1)
# %%

merged_df["mean_score_elementwise"] = merged_df.apply(lambda x: np.mean([ko - reg for ko, reg in zip(x["knockout_logits"], x["region_logits"])]), axis=1)
merged_df["sum_score_elementwise"] = merged_df.apply(lambda x: np.sum([ko - reg for ko, reg in zip(x["knockout_logits"], x["region_logits"])]), axis=1)
# %%
merged_df["mean_score_elementwise_abs"] = merged_df.apply(lambda x: np.mean(np.abs([ko - reg for ko, reg in zip(x["knockout_logits"], x["region_logits"])])), axis=1)
merged_df["sum_score_elementwise_abs"] = merged_df.apply(lambda x: np.sum(np.abs([ko - reg for ko, reg in zip(x["knockout_logits"], x["region_logits"])])), axis=1)
# %%

merged_df.to_feather("/pmglocal/alb2281/repos/get_model/get_model/baselines/hyena/hyena_aggregate_erythro.feather")
# %%

output_df = merged_df[["orig_idx", "chrom", "chromStart", "chromEnd", "mean_score", "sum_score", "mean_score_elementwise", "sum_score_elementwise", "mean_score_elementwise_abs", "sum_score_elementwise_abs"]]
# %%
output_df.to_csv("/pmglocal/alb2281/repos/get_model/get_model/baselines/hyena/hyena_aggregate_erythro.tsv", sep="\t", index=False)
# %%
