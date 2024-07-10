import pandas as pd
import polars as pl
from scipy.sparse import load_npz
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import os

from enformer_pytorch.data import FastaInterval, identity

splits_base_dir = "/pmglocal/alb2281/get/get_rebuttal/data/splits"


def split_data_by_chr(args, resplit=False): 
    splits_dir = os.path.join(splits_base_dir, args.leave_out_chr)
    if not os.path.exists(splits_dir):
        os.makedirs(splits_dir)
    train_path = f"{splits_dir}/train.bed"
    val_path = f"{splits_dir}/val.bed"
    
    if os.path.exists(train_path) and os.path.exists(val_path) and not resplit:
        return train_path, val_path
    else:
        leave_out_chrs = args.leave_out_chr.split(",")   
        bed_df = pd.read_csv(args.atac_data, index_col="index")
        labels = np.array(load_npz(args.labels_path).todense()[:,-1].reshape(-1))[0]
        bed_df["target"] = labels
        train_df = bed_df[~bed_df["Chromosome"].isin(leave_out_chrs)]
        val_df = bed_df[bed_df["Chromosome"].isin(leave_out_chrs)]
        train_df.to_csv(train_path, sep="\t", header=False, index=False)
        val_df.to_csv(val_path, sep="\t", header=False, index=False)
    return train_path, val_path


class GenomeIntervalFinetuneDataset(Dataset):
    def __init__(
        self,
        bed_file,
        fasta_file,
        filter_df_fn = identity,
        chr_bed_to_fasta_map = dict(),
        context_length = None,
        return_seq_indices = False,
        shift_augs = None,
        rc_aug = False,
        return_augs = False
    ):
        super().__init__()
        bed_path = Path(bed_file)
        assert bed_path.exists(), 'path to .bed file must exist'

        df = pl.read_csv(str(bed_path), separator = '\t', has_header = False)
        df = filter_df_fn(df)
        self.df = df
        self.chr_bed_to_fasta_map = chr_bed_to_fasta_map
        self.fasta = FastaInterval(
            fasta_file = fasta_file,
            context_length = context_length,
            return_seq_indices = return_seq_indices,
            shift_augs = shift_augs,
            rc_aug = rc_aug
        )

        self.return_augs = return_augs

    def __len__(self):
        return len(self.df)

    # Modify to return with target for finetuning
    def __getitem__(self, ind):
        interval = self.df.row(ind)
        chr_name, start, end = (interval[0], interval[1], interval[2])
        chr_name = self.chr_bed_to_fasta_map.get(chr_name, chr_name)
        return self.fasta(chr_name, start, end, return_augs = self.return_augs), interval[3]
