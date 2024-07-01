# %%
import pandas as pd 
import json 
import numpy as np
from tqdm import tqdm

# %%
encode_peaks = "/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/data/k562_new_encode_peaks.csv"
encode_df = pd.read_csv(encode_peaks)
preds_dir = "/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/preds/enformer_cage_new_encode_peaks_chr14"
pred_files = os.listdir(preds_dir)
# %%
all_preds = []
for file in pred_files:
    with open(f"{preds_dir}/{file}", "rb") as f:
        preds = np.load(f, allow_pickle=True)
        all_preds.append(preds)
# %
# combine dictionaries together
combined_preds = {}
for batch in all_preds:
    for item in batch:
        for key in item.keys():
            combined_preds[key] = item[key]
# %%
leaveout_df = encode_df.copy()
leaveout_df = leaveout_df.query('Chromosome == "chr14"')

# %%
# save preds to leaveout_df where index is the original index
leaveout_df["preds"] = leaveout_df.index.map(lambda x: combined_preds[x])

# %%
leaveout_df["Expression_sum"] = leaveout_df["Expression_positive"] + leaveout_df["Expression_negative"]

# [4828, 5111]
leaveout_df["mean_preds_4828"] = leaveout_df["preds"].map(lambda x: np.mean(x[0,:]))
leaveout_df["mean_preds_5111"] = leaveout_df["preds"].map(lambda x: np.mean(x[1,:]))
leaveout_df["sum_preds_4828"] = leaveout_df["preds"].map(lambda x: np.sum(x[0,:]))
leaveout_df["sum_preds_5111"] = leaveout_df["preds"].map(lambda x: np.sum(x[1,:]))
leaveout_df["mean_preds_2track_average"] = (leaveout_df["mean_preds_4828"] + leaveout_df["mean_preds_5111"]) / 2
leaveout_df["sum_preds_2track_average"] = (leaveout_df["sum_preds_4828"] + leaveout_df["sum_preds_5111"]) / 2


# %%
from pyranges import PyRanges as pr

# %%
eval_df_without_preds = leaveout_df.drop(columns=["preds"])
enformer_seqs = "/pmglocal/alb2281/get/get_data/enformer_human_sequences.bed"
enformer_df = pd.read_csv(enformer_seqs, sep="\t", header=None)
enformer_df.columns = ["Chromosome", "Start", "End", "Split"]
enformer_df_leaveout = enformer_df.query('Split != "train"')
overlap_bed = pr(eval_df_without_preds.reset_index()).join(pr(enformer_df_leaveout), suffix="_enformer").df[eval_df_without_preds.columns.tolist()+['index']].drop_duplicates()
overlap_bed.set_index("index", inplace=True)

# %%
overlap_bed["preds"] = overlap_bed.index.map(lambda x: combined_preds[x])

# %%
overlap_leaveout_df = overlap_bed.copy()

# %%
overlap_leaveout_df["mean_preds_4828"] = overlap_leaveout_df["preds"].map(lambda x: np.mean(x[0,:]))
overlap_leaveout_df["mean_preds_5111"] = overlap_leaveout_df["preds"].map(lambda x: np.mean(x[1,:]))
overlap_leaveout_df["sum_preds_4828"] = overlap_leaveout_df["preds"].map(lambda x: np.sum(x[0,:]))
overlap_leaveout_df["sum_preds_5111"] = overlap_leaveout_df["preds"].map(lambda x: np.sum(x[1,:]))
overlap_leaveout_df["mean_preds_2track_average"] = (overlap_leaveout_df["mean_preds_4828"] + overlap_leaveout_df["mean_preds_5111"]) / 2
overlap_leaveout_df["sum_preds_4track_average"] = (overlap_leaveout_df["sum_preds_4828"] + overlap_leaveout_df["sum_preds_5111"]) / 2

# %%
leaveout_df_tss_only = leaveout_df.query('TSS > 0')
overlap_leaveout_df_tss = overlap_leaveout_df.query('TSS > 0')


# count > 10
leaveout_df = leaveout_df.query('Count > 10')
leaveout_df_tss_only = leaveout_df_tss_only.query('Count > 10')
overlap_leaveout_df = overlap_leaveout_df.query('Count > 10')
overlap_leaveout_df_tss = overlap_leaveout_df_tss.query('Count > 10')


# %%
eval_df = leaveout_df[["mean_preds_4828", "mean_preds_5111", "mean_preds_2track_average", "sum_preds_4828", "sum_preds_5111", "sum_preds_2track_average", "Expression_sum"]]
eval_df_tss = leaveout_df_tss_only[['mean_preds_4828', 'mean_preds_5111', 'mean_preds_2track_average', 'sum_preds_4828', 'sum_preds_5111', 'sum_preds_2track_average', 'Expression_sum']]
overlap_eval_df = overlap_leaveout_df[["mean_preds_4828", "mean_preds_5111", "mean_preds_2track_average", "sum_preds_4828", "sum_preds_5111", "sum_preds_2track_average", "Expression_sum"]]
overlap_eval_df_tss = overlap_leaveout_df_tss[["mean_preds_4828", "mean_preds_5111", "mean_preds_2track_average", "sum_preds_4828", "sum_preds_5111", "sum_preds_2track_average", "Expression_sum"]]

# %%

pearson_eval_df_corr = eval_df.corr(method='pearson')
pearson_eval_df_tss_corr = eval_df_tss.corr(method='pearson')
pearson_overlap_eval_df_corr = overlap_eval_df.corr(method='pearson')
pearson_overlap_eval_df_tss_corr = overlap_eval_df_tss.corr(method='pearson')
# %%

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

###

spearman_eval_df_corr = eval_df.corr(method='spearman')
spearman_eval_df_tss_corr = eval_df_tss.corr(method='spearman')
spearman_overlap_eval_df_corr = overlap_eval_df.corr(method='spearman')
spearman_overlap_eval_df_tss_corr = overlap_eval_df_tss.corr(method='spearman')
# %%

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


pearson_merged_corr.to_csv("/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/results/enformer_k562_atac_pearson.csv")
spearman_merged_corr.to_csv("/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/results/enformer_k562_atac_spearman.csv")


# %%
import seaborn as sns
sns.scatterplot(data=eval_df_tss, x="sum_preds_2track_average", y="Expression_sum")
# %%

sns.scatterplot(data=, x="sum_preds_4track_average", y="aTPM")
# %%

pearson_merged_corr
# %%
spearman_merged_corr
# %%


## Overlap with count 10 peaks

count_10_peaks = "/pmglocal/alb2281/get/get_data/k562_count_10/k562_count_10.csv"
# %%
count_10_peaks_df = pd.read_csv(count_10_peaks)
# %%
count10_overlap_bed = pr(dnase_peaks_df.reset_index()).join(pr(count_10_peaks_df), suffix="_count10peaks").df[dnase_peaks_df.columns.tolist()+['index']].drop_duplicates()
# %%
count10_overlap_bed.set_index("index", inplace=True)
overlap_leaveout_df = count10_overlap_bed.copy()
#%%

overlap_leaveout_df["preds"] = overlap_leaveout_df.index.map(lambda x: combined_preds[x])

# %%
overlap_leaveout_df["mean_preds_121"] = overlap_leaveout_df["preds"].map(lambda x: np.mean(x[0,:]))
overlap_leaveout_df["mean_preds_122"] = overlap_leaveout_df["preds"].map(lambda x: np.mean(x[1,:]))
overlap_leaveout_df["mean_preds_123"] = overlap_leaveout_df["preds"].map(lambda x: np.mean(x[2,:]))
overlap_leaveout_df["mean_preds_625"] = overlap_leaveout_df["preds"].map(lambda x: np.mean(x[3,:]))
overlap_leaveout_df["sum_preds_121"] = overlap_leaveout_df["preds"].map(lambda x: np.sum(x[0,:]))
overlap_leaveout_df["sum_preds_122"] = overlap_leaveout_df["preds"].map(lambda x: np.sum(x[1,:]))
overlap_leaveout_df["sum_preds_123"] = overlap_leaveout_df["preds"].map(lambda x: np.sum(x[2,:]))
overlap_leaveout_df["sum_preds_625"] = overlap_leaveout_df["preds"].map(lambda x: np.sum(x[3,:]))
overlap_leaveout_df["mean_preds_4track_average"] = (overlap_leaveout_df["mean_preds_121"] + overlap_leaveout_df["mean_preds_122"] + overlap_leaveout_df["mean_preds_123"] + overlap_leaveout_df["mean_preds_625"]) / 4
overlap_leaveout_df["sum_preds_4track_average"] = (overlap_leaveout_df["sum_preds_121"] + overlap_leaveout_df["sum_preds_122"] + overlap_leaveout_df["sum_preds_123"] + overlap_leaveout_df["sum_preds_625"]) / 4

# %%
overlap_leaveout_df = overlap_leaveout_df[['mean_preds_121', 'mean_preds_122', 'mean_preds_123', 'mean_preds_625', 'mean_preds_4track_average', 'sum_preds_121', 'sum_preds_122', 'sum_preds_123', 'sum_preds_625', 'sum_preds_4track_average', 'aTPM']]
# %%
pearson_overlap_df_corr = overlap_leaveout_df.corr(method='pearson')[["aTPM"]]
spearman_overlap_df_corr = overlap_leaveout_df.corr(method='spearman')[["aTPM"]]


# %%
pearson_overlap_df_corr
# %%
spearman_overlap_df_corr
# %%

