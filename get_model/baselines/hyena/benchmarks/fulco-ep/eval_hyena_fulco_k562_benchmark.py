# %%
import json 
import pandas as pd
import numpy as np

# %%
chunk_1_dir = "/burg/pmg/users/alb2281/hyena/results/hyena_chunk_1"
chunk_2_dir = "/burg/pmg/users/alb2281/hyena/results/hyena_chunk_2"
chunk_3_dir = "/burg/pmg/users/alb2281/hyena/results/hyena_chunk_3"
chunk_4_dir = "/burg/pmg/users/alb2281/hyena/results/hyena_chunk_4"
chunk_5_dir = "/burg/pmg/users/alb2281/hyena/results/hyena_leftover"


# %%

chunk_1_files = os.listdir(chunk_1_dir)
chunk_2_files = os.listdir(chunk_2_dir)
chunk_3_files = os.listdir(chunk_3_dir)
chunk_4_files = os.listdir(chunk_4_dir)
chunk_5_files = os.listdir(chunk_5_dir)

# 
# %%
all_preds = []
for file in chunk_1_files:
    with open(os.path.join(chunk_1_dir, file), "r") as f:
        data = json.load(f)
        for d in data:
            all_preds.append(d)
for file in chunk_2_files:
    with open(os.path.join(chunk_2_dir, file), "r") as f:
        data = json.load(f)
        for d in data:
            all_preds.append(d)
for file in chunk_3_files:  
    with open(os.path.join(chunk_3_dir, file), "r") as f:
        data = json.load(f)
        for d in data:
            all_preds.append(d)
for file in chunk_4_files:
    with open(os.path.join(chunk_4_dir, file), "r") as f:
        data = json.load(f)
        for d in data:
            all_preds.append(d)
for file in chunk_5_files:
    with open(os.path.join(chunk_5_dir, file), "r") as f:
        data = json.load(f)
        for d in data:
            all_preds.append(d)

# read list of dicts into dataframe
hyena_df = pd.DataFrame(all_preds)
# %%

hyena_df = hyena_df.drop_duplicates(subset="orig_idx", keep="last")
# %%

hyena_df = hyena_df.sort_values(by="orig_idx")

# %%
fulco_data = "/pmglocal/alb2281/repos/CRISPR_comparison/resources/crispr_data/EPCrisprBenchmark_ensemble_data_GRCh38.tsv"
fulco_df = pd.read_csv(fulco_data, sep="\t")
# %%

# %%
no_start_tss = fulco_df[fulco_df["startTSS"].isna()]
no_start_tss_idx = set(no_start_tss.index)
# %%

# for no startTSS assign empty list to region_logits and knockout_logits
missing_rows = []
for i in no_start_tss_idx:
    missing_rows.append({
        "orig_idx": i,
        "chrom": fulco_df.loc[i, "chrom"],
        "chromStart": fulco_df.loc[i, "chromStart"],
        "chromEnd": fulco_df.loc[i, "chromEnd"],
        "region_logits": [],
        "knockout_logits": []
    })


# %%
missing_df = pd.DataFrame(missing_rows)
# %%
merged_df = pd.concat([hyena_df, missing_df])
# %%

# sort by orig_idx
merged_df = merged_df.sort_values(by="orig_idx")
# %%


merged_df = merged_df.reset_index(drop=True)
# %%

merged_df = merged_df.drop_duplicates(subset="orig_idx", keep="last")
# %%

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
merged_df["mean_score"] = merged_df.apply(compute_mean_score, axis=1)
merged_df["sum_score"] = merged_df.apply(compute_sum_score, axis=1)
# %%

merged_df["mean_score_elementwise"] = merged_df.apply(lambda x: np.mean([ko - reg for ko, reg in zip(x["knockout_logits"], x["region_logits"])]), axis=1)
merged_df["sum_score_elementwise"] = merged_df.apply(lambda x: np.sum([ko - reg for ko, reg in zip(x["knockout_logits"], x["region_logits"])]), axis=1)
# %%
merged_df["mean_score_elementwise_abs"] = merged_df.apply(lambda x: np.mean(np.abs([ko - reg for ko, reg in zip(x["knockout_logits"], x["region_logits"])])), axis=1)
merged_df["sum_score_elementwise_abs"] = merged_df.apply(lambda x: np.sum(np.abs([ko - reg for ko, reg in zip(x["knockout_logits"], x["region_logits"])])), axis=1)
# %%

merged_df.to_feather("/pmglocal/alb2281/repos/get_model/get_model/baselines/hyena/fulco_aggregate.feather")
# %%

output_df = merged_df[["orig_idx", "chrom", "chromStart", "chromEnd", "mean_score", "sum_score", "mean_score_elementwise", "sum_score_elementwise", "mean_score_elementwise_abs", "sum_score_elementwise_abs"]]
# %%
output_df.to_csv("/pmglocal/alb2281/repos/get_model/get_model/baselines/hyena/fulco_aggregate_hyena.tsv", sep="\t", index=False)
# %%
