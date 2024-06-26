# %%
import pandas as pd 
import json 
import numpy as np
from tqdm import tqdm
# %%
preds_dir = "/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/preds/enformer_k562_cage"
# %%
pred_files = os.listdir(preds_dir)
# %%
# sort files by number, exclude "final"
pred_files = [f for f in pred_files if "final" not in f]
pred_files = sorted(pred_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
pred_files.append("enformer_cage_example_final.json")
# %%

leaveout_cage_peaks = "/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/data/cage_peaks_leaveout_k562.csv"
leaveout_df = pd.read_csv(leaveout_cage_peaks)
# %%

all_preds = []
for file in pred_files:
    with open(f"{preds_dir}/{file}", "r") as f:
        preds = json.load(f)
        all_preds.append(preds)

# %%

# combine dictionaries together
combined_preds = {}
for track_idx in all_preds[0].keys():
    combined_preds[track_idx] = {}
    for d in all_preds:
        for k, v in d[track_idx].items():
            combined_preds[track_idx][k] = v
# %%
combined_preds.keys()
# %%
len(combined_preds['4828'].keys())
len(combined_preds['5111'].keys())
# %%
leaveout_df["Expression_sum"] = leaveout_df["Expression_positive"] + leaveout_df["Expression_negative"]
# %%
leaveout_df["Expression_sum"].describe()
#%%
leaveout_df = leaveout_df.query('TSS>0')
# %%
mean_preds_4828 = []
for orig_idx in leaveout_df.index.values:
    mean_preds_4828.append(np.mean(combined_preds['4828'][str(orig_idx)]))

# %%
mean_preds_5111 = []
for orig_idx in leaveout_df.index.values:
    mean_preds_5111.append(np.mean(combined_preds['5111'][str(orig_idx)]))
# %%
mean_preds_4828
# %%
leaveout_df["enformer_track_4828"] = mean_preds_4828
leaveout_df["enformer_track_5111"] = mean_preds_5111
# %%
leaveout_df["enformer_mean"] = (leaveout_df["enformer_track_4828"] + leaveout_df["enformer_track_5111"]) / 2
# %%

# pearson between enformer_mean and expression sum
leaveout_df[["enformer_mean", "Expression_sum"]].corr()
# %%

eval_df = leaveout_df[['Expression_positive', 'Expression_negative', 'enformer_mean', 'aTPM', 'Expression_sum', 'enformer_track_4828', 'enformer_track_5111', 'enformer_mean']]
# %%
eval_df.corr(method='pearson')
# %%
eval_df.corr(method='spearman')
# %%

sum_preds_4828 = []
for orig_idx in leaveout_df.index.values:
    sum_preds_4828.append(np.sum(combined_preds['4828'][str(orig_idx)]))

# %%
eval_df["enformer_track_4828_sum"] = sum_preds_4828
# %%

sum_preds_5111 = []
for orig_idx in range(len(leaveout_df)):
    sum_preds_5111.append(np.sum(combined_preds['5111'][str(orig_idx)]))
# %%

eval_df["enformer_track_5111_sum"] = sum_preds_5111
#%%
max_preds_4828 = []
for orig_idx in leaveout_df.index.values:
    if len(combined_preds['4828'][str(orig_idx)]) == 0:
        max_preds_4828.append(np.nan)
    else:
        max_preds_4828.append(np.array(combined_preds['4828'][str(orig_idx)]).max())

# %%
eval_df["enformer_track_4828_max"] = max_preds_4828
# %%

max_preds_5111 = []
for orig_idx in leaveout_df.index.values:
    if len(combined_preds['5111'][str(orig_idx)]) == 0:
        max_preds_5111.append(np.nan)
    else:
        max_preds_5111.append(np.max(combined_preds['5111'][str(orig_idx)]))
# %%

eval_df["enformer_track_5111_max"] = max_preds_5111
# %%

eval_df.dropna().corr()
# %%
eval_df["enformer_max_mean"] = (eval_df["enformer_track_4828_max"] + eval_df["enformer_track_5111_max"]) / 2
# %%

eval_df.corr(method='spearman')
# %%
eval_df.dropna().plot(x="enformer_max_mean", y="Expression_sum", kind="scatter")
# %%
