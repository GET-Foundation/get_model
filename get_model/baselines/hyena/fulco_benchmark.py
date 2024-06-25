import json
import os
import subprocess
import torch
import re
import pandas as pd
import torch 
from tqdm import tqdm
import numpy as np
from Bio.Seq import reverse_complement, Seq

from transformers import PreTrainedModel
from standalone_hyenadna import HyenaDNAModel
from standalone_hyenadna import CharacterTokenizer
from transformers import AutoModelForCausalLM
import pathlib
import argparse

from atac_rna_data_processing.io.region import Genome


SEQUENCE_LENGTH = 1_000_000
WINDOW_SIZE = 2000
SAVE_EVERY = 100


def single_inference(row, model):
    with torch.inference_mode():
        if row["contains_enhancer"] == False:
            row["region_logits"] = []
            row["knockout_logits"] = []
            output_row = row[["orig_idx", "chrom", "chromStart", "chromEnd", "region_logits", "knockout_logits"]]
        else:
            row["attention_start"] = row["startTSS"] - WINDOW_SIZE // 2
            row["attention_end"] = row["startTSS"] + WINDOW_SIZE // 2

            start_idx = int(row["attention_start"]) - row["region_start"]
            end_idx = int(row["attention_end"]) - row["region_start"]

            input_ids = np.array(tokenizer(row["region_dna_sequence"])["input_ids"])
            region_token_seq = torch.LongTensor(input_ids).unsqueeze(0)  # unsqueeze for batch dim
            region_tok_seq = region_token_seq.to(device)
            region_logits = model(region_tok_seq).logits.cpu().detach().numpy()
            region_logits = region_logits[0, np.arange(start_idx, end_idx), input_ids[start_idx:end_idx]].tolist()  # grab logits corresponding to sequence
            row["region_logits"] = region_logits

            knockout_input_ids = np.array(tokenizer(row["region_dna_sequence_knockout"])["input_ids"])
            knockout_token_seq = torch.LongTensor(knockout_input_ids).unsqueeze(0)  # unsqueeze for batch dim
            knockout_tok_seq = knockout_token_seq.to(device)
            knockout_logits = model(knockout_tok_seq).logits.cpu().detach().numpy()
            knockout_logits = knockout_logits[0, np.arange(start_idx, end_idx), knockout_input_ids[start_idx:end_idx]].tolist()  # grab logits corresponding to sequence
            row["knockout_logits"] = knockout_logits
      
    output_dict = {
        "orig_idx": row["orig_idx"],
        "chrom": row["chrom"],
        "chromStart": row["chromStart"],
        "chromEnd": row["chromEnd"],
        "region_logits": row["region_logits"],
        "knockout_logits": row["knockout_logits"]
    }
    return output_dict


def replace_enhancer_str(region_str, enhancer_str):
    knockout_str = region_str.replace(enhancer_str, "N" * len(enhancer_str))
    return knockout_str


if __name__=="__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--start_idx", type=int, default=0)
    argparse.add_argument("--end_idx", type=int, default=100)
    args = argparse.parse_args()

    fulco_data = "/pmglocal/alb2281/repos/CRISPR_comparison/resources/crispr_data/EPCrisprBenchmark_ensemble_data_GRCh38.tsv"
    fulco_df = pd.read_csv(fulco_data, sep="\t")
    fulco_df["orig_idx"] = fulco_df.index
    fulco_df = fulco_df[fulco_df["startTSS"].notna()]

    genome_path = os.path.join("/manitou/pmg/users/xf2217/interpret_natac/", "hg38.fa")
    genome = Genome('hg38', genome_path)
    
    fulco_df["region_start"] = fulco_df["startTSS"] - SEQUENCE_LENGTH // 2
    fulco_df["region_start"] = fulco_df["region_start"].apply(lambda x: int(x))
    fulco_df["region_end"] = fulco_df["startTSS"] + SEQUENCE_LENGTH // 2
    fulco_df["region_end"] = fulco_df["region_end"].apply(lambda x: int(x))

    fulco_df["enhancer_seq"] = fulco_df.apply(lambda x: genome.get_sequence(x["chrom"], x["chromStart"], x["chromEnd"]).seq, axis=1)
    fulco_df["region_dna_sequence"] = fulco_df.apply(lambda x: genome.get_sequence(x["chrom"], x["region_start"], x["region_end"]).seq, axis=1)
    fulco_df["contains_enhancer"] = fulco_df.apply(lambda x: x["enhancer_seq"] in x["region_dna_sequence"], axis=1)

    fulco_df["region_dna_sequence"] = fulco_df["region_dna_sequence"].apply(lambda x: Seq(x))
    fulco_df["enhancer_seq"] = fulco_df["enhancer_seq"].apply(lambda x: Seq(x))
    # if enhancer string is to the right of the TSS, take the reverse complement of the region and enhancer string (causal Hyena)   
    fulco_df["region_dna_sequence"] = fulco_df.apply(lambda x: str(x["region_dna_sequence"]) if x["chromStart"] < x["startTSS"] else str(x["region_dna_sequence"].reverse_complement()), axis=1)
    fulco_df["enhancer_seq"] = fulco_df.apply(lambda x: str(x["enhancer_seq"]) if x["chromStart"] < x["startTSS"] else str(x["enhancer_seq"].reverse_complement()), axis=1)
    fulco_df["region_dna_sequence_knockout"] = fulco_df.apply(lambda x: replace_enhancer_str(x["region_dna_sequence"], x["enhancer_seq"]), axis=1)

    model = AutoModelForCausalLM.from_pretrained("LongSafari/hyenadna-large-1m-seqlen-hf", trust_remote_code=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    pretrained_model_name = 'hyenadna-large-1m-seqlen'
    max_lengths = {
        'hyenadna-tiny-1k-seqlen': 1024,
        'hyenadna-small-32k-seqlen': 32768,
        'hyenadna-medium-160k-seqlen': 160000,
        'hyenadna-medium-450k-seqlen': 450000,  # T4 up to here
        'hyenadna-large-1m-seqlen': 1_000_000,  # only A100 (paid tier)
    }
    max_length = max_lengths[pretrained_model_name]  # auto selects

    tokenizer = CharacterTokenizer(
        characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
        model_max_length=max_length + 2,  # to account for special tokens, like EOS
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side='left', # since HyenaDNA is causal, we pad on the left
    )

    model.to(device)
    model.eval()

    output_dir = "/pmglocal/alb2281/get/results/hyena-fulco"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    result_col = []
    # iterate over rows of dataframe
    for idx, row in tqdm(fulco_df.iterrows(), total=len(fulco_df)):
        if idx < args.start_idx:
            continue
        if idx >= args.end_idx:
            break
        if idx % SAVE_EVERY == 0:
            # save results to json
            with open(f"hyena_fulco_benchmark_chunk_{idx}.json", "w") as f:
                json.dump(result_col, f)
            result_col = []
        result_col.append(single_inference(row, model))
