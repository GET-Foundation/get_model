# %%
import pandas as pd 
import json 
import numpy as np
from tqdm import tqdm
# %%
cage_peaks = "/burg/pmg/users/alb2281/get/get_data/cage_peaks.csv"

# %%
preds_dir = "/burg/pmg/users/alb2281/enformer/results/enformer_k562_new_encode_chr10,11"
# %%
pred_files = os.listdir(preds_dir)
# %%
# sort files by number, exclude "final"
pred_files = [f for f in pred_files if "final" not in f]
pred_files = sorted(pred_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
pred_files.append("enformer_cage_example_final.json")
# %%

cage_peaks_df = pd.read_csv(cage_peaks)


# %%
cage_peaks_df = cage_peaks_df.query('Chromosome == "chr10" or Chromosome == "chr11"')
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

leaveout_df = cage_peaks_df.copy()
# %%
leaveout_df["Expression_sum"] = leaveout_df["Expression_positive"] + leaveout_df["Expression_negative"]
# %%
leaveout_df["Expression_sum"].describe()
#%%
# leaveout_df = leaveout_df.query('TSS>0')
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


# %%
eval_df = leaveout_df[['Expression_positive', 'Expression_negative', 'enformer_mean', 'aTPM', 'Expression_sum', 'enformer_track_4828', 'enformer_track_5111', 'enformer_mean']]

sum_preds_4828 = []
for orig_idx in leaveout_df.index.values:
    sum_preds_4828.append(np.sum(combined_preds['4828'][str(orig_idx)]))

# %%
eval_df["enformer_track_4828_sum"] = sum_preds_4828
# %%

sum_preds_5111 = []
for orig_idx in leaveout_df.index.values:
    sum_preds_5111.append(np.sum(combined_preds['5111'][str(orig_idx)]))
# %%

eval_df["enformer_track_5111_sum"] = sum_preds_5111
#%%
eval_df["enformer_track_4828_sum_mean"] = (eval_df["enformer_track_4828_sum"] + eval_df["enformer_track_5111_sum"]) / 2

# %%
eval_df.corr(method='pearson')
# %%

# TSS only
leaveout_df_tss = leaveout_df.query('TSS>0')
# %%
mean_preds_4828 = []
for orig_idx in leaveout_df_tss.index.values:
    mean_preds_4828.append(np.mean(combined_preds['4828'][str(orig_idx)]))

# %%
mean_preds_5111 = []
for orig_idx in leaveout_df_tss.index.values:
    mean_preds_5111.append(np.mean(combined_preds['5111'][str(orig_idx)]))
# %%
mean_preds_4828
# %%
leaveout_df_tss["enformer_track_4828"] = mean_preds_4828
leaveout_df_tss["enformer_track_5111"] = mean_preds_5111
# %%
leaveout_df_tss["enformer_mean"] = (leaveout_df["enformer_track_4828"] + leaveout_df["enformer_track_5111"]) / 2
# %%
# %%
eval_df_tss = leaveout_df_tss[['Expression_positive', 'Expression_negative', 'enformer_mean', 'aTPM', 'Expression_sum', 'enformer_track_4828', 'enformer_track_5111', 'enformer_mean']]
# %%

sum_preds_4828 = []
for orig_idx in leaveout_df_tss.index.values:
    sum_preds_4828.append(np.sum(combined_preds['4828'][str(orig_idx)]))

# %%
eval_df_tss["enformer_track_4828_sum"] = sum_preds_4828
# %%

sum_preds_5111 = []
for orig_idx in leaveout_df_tss.index.values:
    sum_preds_5111.append(np.sum(combined_preds['5111'][str(orig_idx)]))
# %%

eval_df_tss["enformer_track_5111_sum"] = sum_preds_5111
#%%

eval_df_tss["enformer_track_4828_sum_mean"] = (eval_df_tss["enformer_track_4828_sum"] + eval_df_tss["enformer_track_5111_sum"]) / 2

eval_df_tss.corr(method='pearson')
# %%

enformer_seqs = "/pmglocal/alb2281/get/get_data/enformer/human/enformer_human_sequences.bed"
# %%
enformer_df = pd.read_csv(enformer_seqs, sep="\t", header=None)
# %%
enformer_df
# %%
enformer_df.columns = ["Chromosome", "Start", "End", "Split"]
# %%

enformer_df_leaveout = enformer_df.query('Split != "train"')
# %%

from pyranges import PyRanges as pr
# %%
overlap_bed = pr(cage_peaks_df.reset_index()).join(pr(enformer_df_leaveout), suffix="_enformer").df[cage_peaks_df.columns.tolist()+['index']].drop_duplicates()

# %%
overlap_bed.set_index("index", inplace=True)

# %%

# %%
overlap_leaveout_df = overlap_bed.copy()
# %%
overlap_leaveout_df["Expression_sum"] = overlap_leaveout_df["Expression_positive"] + overlap_leaveout_df["Expression_negative"]
# %%
overlap_leaveout_df["Expression_sum"].describe()
#%%
# leaveout_df = leaveout_df.query('TSS>0')
# %%
mean_preds_4828 = []
for orig_idx in overlap_leaveout_df.index.values:
    mean_preds_4828.append(np.mean(combined_preds['4828'][str(orig_idx)]))

# %%
mean_preds_5111 = []
for orig_idx in overlap_leaveout_df.index.values:
    mean_preds_5111.append(np.mean(combined_preds['5111'][str(orig_idx)]))
# %%
mean_preds_4828
# %%
overlap_leaveout_df["enformer_track_4828"] = mean_preds_4828
overlap_leaveout_df["enformer_track_5111"] = mean_preds_5111
# %%
overlap_leaveout_df["enformer_mean"] = (overlap_leaveout_df["enformer_track_4828"] + overlap_leaveout_df["enformer_track_5111"]) / 2
# %%


# %%

overlap_eval_df = overlap_leaveout_df[['Expression_positive', 'Expression_negative', 'enformer_mean', 'aTPM', 'Expression_sum', 'enformer_track_4828', 'enformer_track_5111', 'enformer_mean']]
# %%
sum_preds_4828 = []
for orig_idx in overlap_leaveout_df.index.values:
    sum_preds_4828.append(np.sum(combined_preds['4828'][str(orig_idx)]))

# %%
overlap_eval_df["enformer_track_4828_sum"] = sum_preds_4828
# %%

sum_preds_5111 = []
for orig_idx in overlap_leaveout_df.index.values:
    sum_preds_5111.append(np.sum(combined_preds['5111'][str(orig_idx)]))
# %%

overlap_eval_df["enformer_track_5111_sum"] = sum_preds_5111
#%%

overlap_eval_df["enformer_track_4828_sum_mean"] = (overlap_eval_df["enformer_track_4828_sum"] + overlap_eval_df["enformer_track_5111_sum"]) / 2

overlap_eval_df.corr(method='pearson')
# %%

# TSS only
overlap_leaveout_df_tss = overlap_leaveout_df.query('TSS>0')
# %%
mean_preds_4828 = []
for orig_idx in overlap_leaveout_df_tss.index.values:
    mean_preds_4828.append(np.mean(combined_preds['4828'][str(orig_idx)]))

# %%
mean_preds_5111 = []
for orig_idx in overlap_leaveout_df_tss.index.values:
    mean_preds_5111.append(np.mean(combined_preds['5111'][str(orig_idx)]))
# %%
mean_preds_4828
# %%
overlap_leaveout_df_tss["enformer_track_4828"] = mean_preds_4828
overlap_leaveout_df_tss["enformer_track_5111"] = mean_preds_5111
# %%
overlap_leaveout_df_tss["enformer_mean"] = (overlap_leaveout_df_tss["enformer_track_4828"] + overlap_leaveout_df_tss["enformer_track_5111"]) / 2
# %%

overlap_eval_df_tss = overlap_leaveout_df_tss[['Expression_positive', 'Expression_negative', 'enformer_mean', 'aTPM', 'Expression_sum', 'enformer_track_4828', 'enformer_track_5111', 'enformer_mean']]
# %%
sum_preds_4828 = []
for orig_idx in overlap_leaveout_df_tss.index.values:
    sum_preds_4828.append(np.sum(combined_preds['4828'][str(orig_idx)]))

# %%
overlap_eval_df_tss["enformer_track_4828_sum"] = sum_preds_4828
# %%

sum_preds_5111 = []
for orig_idx in overlap_leaveout_df_tss.index.values:
    sum_preds_5111.append(np.sum(combined_preds['5111'][str(orig_idx)]))
# %%

overlap_eval_df_tss["enformer_track_5111_sum"] = sum_preds_5111
#%%

overlap_eval_df_tss["enformer_track_4828_sum_mean"] = (overlap_eval_df_tss["enformer_track_4828_sum"] + overlap_eval_df_tss["enformer_track_5111_sum"]) / 2

overlap_eval_df_tss.corr(method='pearson')
# %%

pearson_eval_df_corr = eval_df.corr(method='pearson')
pearson_eval_df_tss_corr = eval_df_tss.corr(method='pearson')
pearson_overlap_eval_df_corr = overlap_eval_df.corr(method='pearson')
pearson_overlap_eval_df_tss_corr = overlap_eval_df_tss.corr(method='pearson')
# %%

# subset each df to column Expression sum
pearson_eval_df_corr = pearson_eval_df_corr[["Expression_sum"]]
pearson_eval_df_tss_corr = pearson_eval_df_tss_corr[["Expression_sum"]]
pearson_overlap_eval_df_corr = pearson_overlap_eval_df_corr[["Expression_sum"]]
pearson_overlap_eval_df_tss_corr = pearson_overlap_eval_df_tss_corr[["Expression_sum"]]
# %%

pearson_merged_corr = pd.concat([pearson_eval_df_corr, pearson_eval_df_tss_corr, pearson_overlap_eval_df_corr, pearson_overlap_eval_df_tss_corr], axis=1)
# %%

# rename columns
pearson_merged_corr.columns = ["all_peaks", "all_peaks_tss", "basenji_test", "basenji_test_tss"]
# %%

# select rows
pearson_merged_corr = pearson_merged_corr.loc[["enformer_mean", "aTPM", "enformer_track_4828", "enformer_track_5111", "enformer_mean", "enformer_track_4828_sum", "enformer_track_5111_sum", "enformer_track_4828_sum_mean"]]
# %%

# rename rows
pearson_merged_corr = pearson_merged_corr.drop_duplicates()
# %%
pearson_merged_corr = pearson_merged_corr.drop_duplicates()
# %%

# drop first row
pearson_merged_corr = pearson_merged_corr.iloc[1:]
# %%

# rename index
pearson_merged_corr.index = ["enformer_track_mean_2track_avg", "aTPM", "enformer_track_4828_mean", "enformer_track_5111_mean", "enformer_track_4828_sum", "enformer_track_5111_sum", "enformer_track_sum_2track_avg"]
# %%

# reorder row
pearson_merged_corr = pearson_merged_corr.reindex(["enformer_track_4828_mean", "enformer_track_5111_mean", "enformer_track_mean_2track_avg", "enformer_track_4828_sum", "enformer_track_5111_sum", "enformer_track_sum_2track_avg", "aTPM"])
# %%
pearson_merged_corr.to_csv("/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/results/enformer_k562_cage_pearson.csv")
# %%


###

spearman_eval_df_corr = eval_df.corr(method='spearman')
spearman_eval_df_tss_corr = eval_df_tss.corr(method='spearman')
spearman_overlap_eval_df_corr = overlap_eval_df.corr(method='spearman')
spearman_overlap_eval_df_tss_corr = overlap_eval_df_tss.corr(method='spearman')
# %%

# subset each df to column Expression sum
spearman_eval_df_corr = spearman_eval_df_corr[["Expression_sum"]]
spearman_eval_df_tss_corr = spearman_eval_df_tss_corr[["Expression_sum"]]
spearman_overlap_eval_df_corr = spearman_overlap_eval_df_corr[["Expression_sum"]]
spearman_overlap_eval_df_tss_corr = spearman_overlap_eval_df_tss_corr[["Expression_sum"]]
# %%

spearman_merged_corr = pd.concat([spearman_eval_df_corr, spearman_eval_df_tss_corr, spearman_overlap_eval_df_corr, spearman_overlap_eval_df_tss_corr], axis=1)
# %%

# rename columns
spearman_merged_corr.columns = ["all_peaks", "all_peaks_tss", "basenji_test", "basenji_test_tss"]
# %%

# select rows
spearman_merged_corr = spearman_merged_corr.loc[["enformer_mean", "aTPM", "enformer_track_4828", "enformer_track_5111", "enformer_mean", "enformer_track_4828_sum", "enformer_track_5111_sum", "enformer_track_4828_sum_mean"]]
# %%

# rename rows
spearman_merged_corr = spearman_merged_corr.drop_duplicates()
# %%
# %%
# rename index
spearman_merged_corr.index = ["enformer_track_mean_2track_avg", "aTPM", "enformer_track_4828_mean", "enformer_track_5111_mean", "enformer_track_4828_sum", "enformer_track_5111_sum", "enformer_track_sum_2track_avg"]
# %%

# reorder row
spearman_merged_corr = spearman_merged_corr.reindex(["enformer_track_4828_mean", "enformer_track_5111_mean", "enformer_track_mean_2track_avg", "enformer_track_4828_sum", "enformer_track_5111_sum", "enformer_track_sum_2track_avg", "aTPM"])
# %%
spearman_merged_corr.to_csv("/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/results/enformer_k562_cage_spearman.csv")
# %%
