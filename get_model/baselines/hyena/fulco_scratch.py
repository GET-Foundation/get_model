# %%
import pandas as pd
from atac_rna_data_processing.io.region import Genome
import os 

# %%
SEQUENCE_LENGTH = 1_000_000

fulco_data = "/pmglocal/alb2281/repos/CRISPR_comparison/resources/crispr_data/EPCrisprBenchmark_ensemble_data_GRCh38.tsv"
fulco_df = pd.read_csv(fulco_data, sep="\t")
fulco_df["orig_idx"] = fulco_df.index
fulco_df = fulco_df[fulco_df["startTSS"].notna()]

genome_path = os.path.join("/manitou/pmg/users/xf2217/interpret_natac/", "hg38.fa")
genome = Genome('hg38', genome_path)

fulco_df["enhancer_seq"] = fulco_df.apply(lambda x: genome.get_sequence(x["chrom"], x["chromStart"], x["chromEnd"]).seq, axis=1)
fulco_df["region_start"] = fulco_df["startTSS"] - SEQUENCE_LENGTH // 2
fulco_df["region_start"] = fulco_df["region_start"].apply(lambda x: int(x))
fulco_df["region_end"] = fulco_df["startTSS"] + SEQUENCE_LENGTH // 2
fulco_df["region_end"] = fulco_df["region_end"].apply(lambda x: int(x))
fulco_df["region_dna_sequence"] = fulco_df.apply(lambda x: genome.get_sequence(x["chrom"], x["region_start"], x["region_end"]).seq, axis=1)
# %%
fulco_df["overlap"] = fulco_df.apply(lambda x: max(0, min(x["chromEnd"], x["region_end"]) - max(x["chromStart"], x["region_start"])), axis=1)
# %%