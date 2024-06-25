# %%
import json 
import pandas as pd

# %%
chunk_1_dir = "/burg/pmg/users/alb2281/hyena/results/hyena_chunk_1"
chunk_2_dir = "/burg/pmg/users/alb2281/hyena/results/hyena_chunk_2"
chunk_3_dir = "/burg/pmg/users/alb2281/hyena/results/hyena_chunk_3"
chunk_4_dir = "/burg/pmg/users/alb2281/hyena/results/hyena_chunk_4"

# %%

chunk_1_files = os.listdir(chunk_1_dir)
chunk_2_files = os.listdir(chunk_2_dir)
chunk_3_files = os.listdir(chunk_3_dir)
chunk_4_files = os.listdir(chunk_4_dir)
# %%

# sort by file number
chunk_1_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
chunk_2_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
chunk_3_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
chunk_4_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
# %%

all_preds = []
for file in chunk_1_files:
    with open(os.path.join(chunk_1_dir, file), "r") as f:
        data = json.load(f)
        for d in data:
            all_preds.append(d)
# %%
len(all_preds)
# %%
for file in chunk_2_files:
    with open(os.path.join(chunk_2_dir, file), "r") as f:
        data = json.load(f)
        for d in data:
            all_preds.append(d)

# %%
len(all_preds)
# %%
for file in chunk_3_files:
    with open(os.path.join(chunk_3_dir, file), "r") as f:
        data = json.load(f)
        for d in data:
            all_preds.append(d)
# %%
for file in chunk_4_files:
    with open(os.path.join(chunk_4_dir, file), "r") as f:
        data = json.load(f)
        for d in data:
            all_preds.append(d)
# %%
len(all_preds)
# %%

# read list of dicts into dataframe
hyena_df = pd.DataFrame(all_preds)
# %%

hyena_df = hyena_df.drop_duplicates(subset="orig_idx", keep="last")
# %%

# %%
fulco_data = "/pmglocal/alb2281/repos/CRISPR_comparison/resources/crispr_data/EPCrisprBenchmark_ensemble_data_GRCh38.tsv"
fulco_df = pd.read_csv(fulco_data, sep="\t")
# %%

# %%
no_start_tss = fulco_df[fulco_df["startTSS"].isna()]
no_start_tss_idx = set(no_start_tss.index)
# %%

missing = set(range(0, len(fulco_df))) - set(hyena_df["orig_idx"]) - no_start_tss_idx
# %%
with open("missing_indices.txt", "w") as f:
    for i in missing:
        f.write(f"{i}\n")

# %%
hyena_df
# %%
fulco_df
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

rows_with_empty_region_logits = merged_df[merged_df["region_logits"].apply(lambda x: len(x) == 0)]
# %%

leftover_dir = "/pmglocal/alb2281/get/results/hyena-fulco/leftover"
leftover_files = os.listdir(leftover_dir)

leftover_entries = []

for file in leftover_files:
    with open(os.path.join(leftover_dir, file), "r") as f:
        data = json.load(f)
        for d in data:
            leftover_entries.append(d)

leftover_df = pd.DataFrame(leftover_entries)
merged_hyena_df = pd.concat([merged_df, leftover_df])
merged_hyena_df = merged_hyena_df.sort_values(by="orig_idx")
# %%

merged_hyena_df = merged_hyena_df.drop_duplicates(subset="orig_idx", keep="last")
# %%

missing_idx = set(range(0, len(fulco_df))) - set(merged_hyena_df["orig_idx"])
# %%

with open("missing_indices.txt", "w") as f:
    for i in missing_idx:
        f.write(f"{i}\n")
# %%
