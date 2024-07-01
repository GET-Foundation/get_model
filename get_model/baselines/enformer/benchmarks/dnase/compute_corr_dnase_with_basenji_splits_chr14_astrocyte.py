# %%
import pandas as pd 
import json 
import numpy as np
from tqdm import tqdm
# %%
astrocyte_peaks = "/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/data/118.csv"
astrocyte_df = pd.read_csv(astrocyte_peaks)


#%% 
preds_dir = "/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/preds/enformer_dnase_preds_astrocyte"
pred_files = os.listdir(preds_dir)

names=["Chromosome", "Start", "End", "aTPM"]
atac_peaks = pd.read_csv("/manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/fetal_adult/118.atac.bed", names=names, sep="\t")


# %%
astrocyte_df = astrocyte_df.merge(atac_peaks, on=["Chromosome", "Start", "End"], how="inner")


# %%

all_preds = []
for file in pred_files:
    with open(f"{preds_dir}/{file}", "rb") as f:
        preds = np.load(f, allow_pickle=True)
        all_preds.append(preds)
# %
# %%

# combine dictionaries together
combined_preds = {}
for batch in all_preds:
    for item in batch:
        for key in item.keys():
            combined_preds[key] = item[key]
# %%

leaveout_df = astrocyte_df.copy()

# %%
# save preds to leaveout_df where index is the original index
leaveout_df["preds"] = leaveout_df.index.map(lambda x: combined_preds[x])

# %%
leaveout_df["aTPM"].describe()
#%%
# leaveout_df = leaveout_df.query('TSS>0')
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
# leaveout_df_tss_only = leaveout_df.query('TSS>0')

# %%

from pyranges import PyRanges as pr

# %%

enformer_seqs = "/pmglocal/alb2281/get/get_data/enformer_human_sequences.bed"
enformer_df = pd.read_csv(enformer_seqs, sep="\t", header=None)
enformer_df.columns = ["Chromosome", "Start", "End", "Split"]
enformer_df_leaveout = enformer_df.query('Split != "train"')
overlap_bed = pr(astrocyte_df.reset_index()).join(pr(enformer_df_leaveout), suffix="_enformer").df[astrocyte_df.columns.tolist()+['index']].drop_duplicates()
overlap_bed.set_index("index", inplace=True)
overlap_leaveout_df = overlap_bed.copy().query('Chromosome == "chr13"')
#%%
# get distance in enformer_df
enformer_df_copy = enformer_df.copy()
enformer_df_copy["Distance"] = enformer_df_copy["End"] - enformer_df_copy["Start"]
total_nt = enformer_df_copy.groupby('Chromosome').Distance.sum()
test_nt = enformer_df_copy.query('Split != "train"').groupby('Chromosome').Distance.sum()
train_nt = enformer_df_copy.query('Split == "train"').groupby('Chromosome').Distance.sum()
join_nt = test_nt.to_frame().join(train_nt, lsuffix="_test", rsuffix="_train")
join_nt = join_nt.join(total_nt, rsuffix="_total")
join_nt['test_frac'] = join_nt['Distance_test'] / join_nt['Distance']
join_nt['train_frac'] = join_nt['Distance_train'] / join_nt['Distance']
join_nt
#%%

overlap_leaveout_df["preds"] = overlap_leaveout_df.index.map(lambda x: combined_preds[x])

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
overlap_leaveout_df_tss = overlap_leaveout_df.query('TSS>0')

# %%
eval_df = leaveout_df[["mean_preds_76", "mean_preds_77", "mean_preds_78", "mean_preds_133", "mean_preds_4track_average", "sum_preds_76", "sum_preds_77", "sum_preds_78", "sum_preds_133", "sum_preds_4track_average", "aTPM"]]
# eval_df_tss = leaveout_df_tss_only[["mean_preds_76", "mean_preds_77", "mean_preds_78", "mean_preds_133", "mean_preds_4track_average", "sum_preds_76", "sum_preds_77", "sum_preds_78", "sum_preds_133", "sum_preds_4track_average", "aTPM"]]
overlap_eval_df = overlap_leaveout_df[["mean_preds_76", "mean_preds_77", "mean_preds_78", "mean_preds_133", "mean_preds_4track_average", "sum_preds_76", "sum_preds_77", "sum_preds_78", "sum_preds_133", "sum_preds_4track_average", "aTPM"]]
# overlap_eval_df_tss = overlap_leaveout_df_tss[["mean_preds_76", "mean_preds_77", "mean_preds_78", "mean_preds_133", "mean_preds_4track_average", "sum_preds_76", "sum_preds_77", "sum_preds_78", "sum_preds_133", "sum_preds_4track_average", "aTPM"]]

# %%

pearson_eval_df_corr = eval_df.corr(method='pearson')
# pearson_eval_df_tss_corr = eval_df_tss.corr(method='pearson')
pearson_overlap_eval_df_corr = overlap_eval_df.corr(method='pearson')
# pearson_overlap_eval_df_tss_corr = overlap_eval_df_tss.corr(method='pearson')
# %%

pearson_eval_df_corr = pearson_eval_df_corr[["aTPM"]]
# pearson_eval_df_tss_corr = pearson_eval_df_tss_corr[["aTPM"]]
pearson_overlap_eval_df_corr = pearson_overlap_eval_df_corr[["aTPM"]]
# pearson_overlap_eval_df_tss_corr = pearson_overlap_eval_df_tss_corr[["aTPM"]]
# %%

pearson_merged_corr = pd.concat([pearson_eval_df_corr, pearson_overlap_eval_df_corr], axis=1)
# %%

# rename columns
pearson_merged_corr.columns = ["all_peaks", "basenji_test"]
# %%

# select rows

###

spearman_eval_df_corr = eval_df.corr(method='spearman')
# spearman_eval_df_tss_corr = eval_df_tss.corr(method='spearman')
spearman_overlap_eval_df_corr = overlap_eval_df.corr(method='spearman')
# spearman_overlap_eval_df_tss_corr = overlap_eval_df_tss.corr(method='spearman')
# %%

spearman_eval_df_corr = spearman_eval_df_corr[["aTPM"]]
# spearman_eval_df_tss_corr = spearman_eval_df_tss_corr[["aTPM"]]
spearman_overlap_eval_df_corr = spearman_overlap_eval_df_corr[["aTPM"]]
# spearman_overlap_eval_df_tss_corr = spearman_overlap_eval_df_tss_corr[["aTPM"]]
# %%

spearman_merged_corr = pd.concat([spearman_eval_df_corr, spearman_overlap_eval_df_corr], axis=1)
# %%

# rename columns
spearman_merged_corr.columns = ["all_peaks", "basenji_test"]
# %%


pearson_merged_corr.to_csv("/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/results/enformer_k562_atac_pearson.csv")
spearman_merged_corr.to_csv("/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/results/enformer_k562_atac_spearman.csv")


# %%
import seaborn as sns
sns.scatterplot(data=eval_df, x="sum_preds_4track_average", y="aTPM")
# %%

sns.scatterplot(data=overlap_eval_df, x="sum_preds_4track_average", y="aTPM")
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

