# %%
import pandas as pd 
import json 
import numpy as np
from tqdm import tqdm
# %%
astrocyte_peaks = "/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/data/118.csv"
astrocyte_df = pd.read_csv(astrocyte_peaks)
preds_dir = "/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/preds/enformer_dnase_preds_astrocyte"
pred_files = os.listdir(preds_dir)

names=["Chromosome", "Start", "End", "aTPM"]
atac_peaks = pd.read_csv("/manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/fetal_adult/118.atac.bed", names=names, sep="\t")
astrocyte_df = astrocyte_df.merge(atac_peaks, on=["Chromosome", "Start", "End"], how="inner")
astrocyte_tss = np.load("/manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/fetal_adult/118.tss.npy")


# %%
# store 1 in TSS column if either index of astrocyte_tss is True 
tss_df = pd.DataFrame(astrocyte_tss, columns=["TSS_plus", "TSS_minus"])
tss_df["TSS"] = tss_df["TSS_plus"] | tss_df["TSS_minus"]
# convert to 0  if False and 1 if True
tss_df["TSS"] = tss_df["TSS"].map(lambda x: 1 if x else 0)
astrocyte_df["TSS"] = tss_df["TSS"]


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

leaveout_df = astrocyte_df.copy()
leaveout_df["preds"] = leaveout_df.index.map(lambda x: combined_preds[x])

# %%
# [76, 77, 78, 133]
leaveout_df["mean_preds_76"] = leaveout_df["preds"].map(lambda x: np.mean(x[0,:]))
leaveout_df["mean_preds_77"] = leaveout_df["preds"].map(lambda x: np.mean(x[1,:]))
leaveout_df["mean_preds_78"] = leaveout_df["preds"].map(lambda x: np.mean(x[2,:]))
leaveout_df["mean_preds_133"] = leaveout_df["preds"].map(lambda x: np.mean(x[3,:]))
leaveout_df["sum_preds_76"] = leaveout_df["preds"].map(lambda x: np.sum(x[0,:]))
leaveout_df["sum_preds_77"] = leaveout_df["preds"].map(lambda x: np.sum(x[1,:]))
leaveout_df["sum_preds_78"] = leaveout_df["preds"].map(lambda x: np.sum(x[2,:]))
leaveout_df["sum_preds_133"] = leaveout_df["preds"].map(lambda x: np.sum(x[3,:]))
leaveout_df["mean_preds_4track_average"] = (leaveout_df["mean_preds_76"] + leaveout_df["mean_preds_77"] + leaveout_df["mean_preds_78"] + leaveout_df["mean_preds_133"]) / 4
leaveout_df["sum_preds_4track_average"] = (leaveout_df["sum_preds_76"] + leaveout_df["sum_preds_77"] + leaveout_df["sum_preds_78"] + leaveout_df["sum_preds_133"]) / 4

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
overlap_leaveout_df["mean_preds_76"] = overlap_leaveout_df["preds"].map(lambda x: np.mean(x[0,:]))
overlap_leaveout_df["mean_preds_77"] = overlap_leaveout_df["preds"].map(lambda x: np.mean(x[1,:]))
overlap_leaveout_df["mean_preds_78"] = overlap_leaveout_df["preds"].map(lambda x: np.mean(x[2,:]))
overlap_leaveout_df["mean_preds_133"] = overlap_leaveout_df["preds"].map(lambda x: np.mean(x[3,:]))
overlap_leaveout_df["sum_preds_76"] = overlap_leaveout_df["preds"].map(lambda x: np.sum(x[0,:]))
overlap_leaveout_df["sum_preds_77"] = overlap_leaveout_df["preds"].map(lambda x: np.sum(x[1,:]))
overlap_leaveout_df["sum_preds_78"] = overlap_leaveout_df["preds"].map(lambda x: np.sum(x[2,:]))
overlap_leaveout_df["sum_preds_133"] = overlap_leaveout_df["preds"].map(lambda x: np.sum(x[3,:]))
overlap_leaveout_df["mean_preds_4track_average"] = (overlap_leaveout_df["mean_preds_76"] + overlap_leaveout_df["mean_preds_77"] + overlap_leaveout_df["mean_preds_78"] + overlap_leaveout_df["mean_preds_133"]) / 4
overlap_leaveout_df["sum_preds_4track_average"] = (overlap_leaveout_df["sum_preds_76"] + overlap_leaveout_df["sum_preds_77"] + overlap_leaveout_df["sum_preds_78"] + overlap_leaveout_df["sum_preds_133"]) / 4

# %%
leaveout_df_tss_only = leaveout_df.query('TSS > 0')
overlap_leaveout_df_tss = overlap_leaveout_df.query('TSS > 0')


# count > 10
leaveout_df = leaveout_df.query('Count > 10')
leaveout_df_tss_only = leaveout_df_tss_only.query('Count > 10')
overlap_leaveout_df = overlap_leaveout_df.query('Count > 10')
overlap_leaveout_df_tss = overlap_leaveout_df_tss.query('Count > 10')


# %%
eval_df = leaveout_df[["mean_preds_76", "mean_preds_77", "mean_preds_78", "mean_preds_133", "mean_preds_4track_average", "sum_preds_76", "sum_preds_77", "sum_preds_78", "sum_preds_133", "sum_preds_4track_average", "aTPM"]]
eval_df_tss = leaveout_df_tss_only[["mean_preds_76", "mean_preds_77", "mean_preds_78", "mean_preds_133", "mean_preds_4track_average", "sum_preds_76", "sum_preds_77", "sum_preds_78", "sum_preds_133", "sum_preds_4track_average", "aTPM"]]
overlap_eval_df = overlap_leaveout_df[["mean_preds_76", "mean_preds_77", "mean_preds_78", "mean_preds_133", "mean_preds_4track_average", "sum_preds_76", "sum_preds_77", "sum_preds_78", "sum_preds_133", "sum_preds_4track_average", "aTPM"]]
overlap_eval_df_tss = overlap_leaveout_df_tss[["mean_preds_76", "mean_preds_77", "mean_preds_78", "mean_preds_133", "mean_preds_4track_average", "sum_preds_76", "sum_preds_77", "sum_preds_78", "sum_preds_133", "sum_preds_4track_average", "aTPM"]]

# %%

pearson_eval_df_corr = eval_df.corr(method='pearson')
pearson_eval_df_tss_corr = eval_df_tss.corr(method='pearson')
pearson_overlap_eval_df_corr = overlap_eval_df.corr(method='pearson')
pearson_overlap_eval_df_tss_corr = overlap_eval_df_tss.corr(method='pearson')
# %%

pearson_eval_df_corr = pearson_eval_df_corr[["aTPM"]]
pearson_eval_df_tss_corr = pearson_eval_df_tss_corr[["aTPM"]]
pearson_overlap_eval_df_corr = pearson_overlap_eval_df_corr[["aTPM"]]
pearson_overlap_eval_df_tss_corr = pearson_overlap_eval_df_tss_corr[["aTPM"]]
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

spearman_eval_df_corr = spearman_eval_df_corr[["aTPM"]]
spearman_eval_df_tss_corr = spearman_eval_df_tss_corr[["aTPM"]]
spearman_overlap_eval_df_corr = spearman_overlap_eval_df_corr[["aTPM"]]
spearman_overlap_eval_df_tss_corr = spearman_overlap_eval_df_tss_corr[["aTPM"]]
# %%

spearman_merged_corr = pd.concat([spearman_eval_df_corr, spearman_eval_df_tss_corr, spearman_overlap_eval_df_corr, spearman_overlap_eval_df_tss_corr], axis=1)
# %%

# rename columns
spearman_merged_corr.columns = ["all_peaks", "all_peaks_tss", "basenji_test", "basenji_test_tss"]
# %%

pearson_merged_corr
# %%
spearman_merged_corr
# %%
