# %%
import pandas as pd

# %%
fulco_data = "/pmglocal/alb2281/repos/CRISPR_comparison/resources/crispr_data/EPCrisprBenchmark_ensemble_data_GRCh38.tsv"
# %%
fulco_df = pd.read_csv(fulco_data, sep="\t")
# %%

# add new column with 0-1 label for boolean Regulated column
fulco_df["Label"] = fulco_df["Regulated"].apply(lambda x: 1 if x == True else 0)
# %%

SEQUENCE_LENGTH = 393216


# %%
fulco_df["region_start"] = fulco_df["startTSS"] - SEQUENCE_LENGTH/2
# %%
fulco_df["region_end"] = fulco_df["startTSS"] + SEQUENCE_LENGTH/2
# %%



