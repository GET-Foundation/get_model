import torch
import polars as pl
from enformer_pytorch import Enformer, GenomeIntervalDataset
import argparse

from enformer_pytorch import from_pretrained
from enformer_pytorch.finetune import HeadAdapterWrapper

from utils import *


device = "cuda"
hg38_path = "/pmglocal/alb2281/get/get_data/hg38.ml.fa"


def main(args):
    train_path, val_path = split_data_by_chr(args)
    enformer = from_pretrained('EleutherAI/enformer-official-rough', target_length=2, use_tf_gamma=False) # target_length=1 throws error
    model = HeadAdapterWrapper(
        enformer = enformer,
        num_tracks = 1,
        post_transformer_embed = False,
        auto_set_target_length = False, # Override infer target_length from target dimensions
    ).to(device)
    model.to(device)

    dataset_train = GenomeIntervalFinetuneDataset(
        bed_file = train_path,                         
        fasta_file = hg38_path,                   
        return_seq_indices = True,                          
        shift_augs = (-2, 2),                              
        context_length = 196_608,
    )
    dataset_val = GenomeIntervalFinetuneDataset(
        bed_file = val_path,                         
        fasta_file = hg38_path,                   
        return_seq_indices = True,                          
        shift_augs = (-2, 2),                              
        context_length = 196_608,
    )

    breakpoint()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--atac_data", type=str, required=True)
    parser.add_argument("--labels_path", type=str, required=True)
    parser.add_argument("--leave_out_chr", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    main(args)
