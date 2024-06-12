# %%
import os
import numpy as np
import json
from tqdm import tqdm
import pandas as pd

# %%
preds_dir = "/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/preds/k562_fulco_benchmark_per_nt_norm"

# %%
# iterate over prediction files and calculate correlation with ground truth
files = os.listdir(preds_dir)

# %%
# sort files by number
files = sorted(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))

# %%
pred_col = []
for file in tqdm(files):
    with open(os.path.join(preds_dir, file), "r") as f:
        preds = json.load(f)
        for pred in preds:
            fulco_index = pred[0]
            if pred[1] is None:
                pred_col.append([fulco_index, None, None])
            else:
                pred_window = np.sum(np.abs(pred[1]))
                background = pred[2]
                mean_background = np.mean(np.abs(background))
                pred_col.append([fulco_index, pred_window, mean_background])
            
# %%

for idx, pred in enumerate(pred_col):
    assert(idx == pred[0])

# %%
fulco_data = "/pmglocal/alb2281/repos/CRISPR_comparison/resources/crispr_data/EPCrisprBenchmark_ensemble_data_GRCh38.tsv"
fulco_data = pd.read_csv(fulco_data, sep="\t")
# %%

fulco_data
# %%
pred_df = pd.DataFrame(pred_col, columns=["fulco_index", "pred_window", "mean_background"])
# %%
pred_df["pred_norm"] = pred_df["pred_window"] / pred_df["mean_background"]
# %%
fulco_data["pred_norm"] = pred_df["pred_norm"]
# %%
names = ["chr", "start", "end", "TargetGene", "CellType", "enformer_background_norm"]

# %%
enformer_df = fulco_data[["chrom", "chromStart", "chromEnd", "measuredGeneSymbol", "CellType", "pred_norm"]]
# %%
enformer_df.to_csv("/pmglocal/alb2281/repos/CRISPR_comparison/resources/example/enformer_norm.tsv", sep="\t", index=False, header=True)
# %%
