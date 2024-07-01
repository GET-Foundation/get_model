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
SAVE_EVERY = 50


def single_inference(row, model):
    with torch.inference_mode():
        if row["contains_enhancer"] == False:
            row["region_logits"] = []
            row["knockout_logits"] = []
        else:
            row["attention_start"] = row["TSS"] - WINDOW_SIZE // 2
            row["attention_end"] = row["TSS"] + WINDOW_SIZE // 2

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
        "chrom": row["Chromosome"],
        "chromStart": row["Start"],
        "chromEnd": row["End"],
        "region_logits": row["region_logits"],
        "knockout_logits": row["knockout_logits"]
    }
    return output_dict


def replace_enhancer_str(region_str, enhancer_str):
    knockout_str = region_str.replace(enhancer_str, "N" * len(enhancer_str))
    return knockout_str


if __name__=="__main__":
    argparse = argparse.ArgumentParser()
    args = argparse.parse_args()

    data = "/burg/pmg/users/alb2281/hyena/data/eryth_enformer.csv"
    data_df = pd.read_csv(data, sep="\t")
    data_df["orig_idx"] = data_df.index

    genome_path = os.path.join("/manitou/pmg/users/xf2217/interpret_natac/", "hg38.fa")
    genome = Genome('hg38', genome_path)
    
    data_df["region_start"] = data_df["TSS"] - SEQUENCE_LENGTH // 2
    data_df["region_start"] = data_df["region_start"].apply(lambda x: int(x))
    data_df["region_end"] = data_df["TSS"] + SEQUENCE_LENGTH // 2
    data_df["region_end"] = data_df["region_end"].apply(lambda x: int(x))

    data_df["enhancer_seq"] = data_df.apply(lambda x: genome.get_sequence(x["Chromosome"], x["Start"], x["End"]).seq, axis=1)
    data_df["region_dna_sequence"] = data_df.apply(lambda x: genome.get_sequence(x["Chromosome"], x["region_start"], x["region_end"]).seq, axis=1)
    data_df["contains_enhancer"] = data_df.apply(lambda x: x["enhancer_seq"] in x["region_dna_sequence"], axis=1)

    data_df["region_dna_sequence"] = data_df["region_dna_sequence"].apply(lambda x: Seq(x))
    data_df["enhancer_seq"] = data_df["enhancer_seq"].apply(lambda x: Seq(x))
    # if enhancer string is to the right of the TSS, take the reverse complement of the region and enhancer string (causal Hyena)   
    data_df["region_dna_sequence"] = data_df.apply(lambda x: str(x["region_dna_sequence"]) if x["Start"] < x["TSS"] else str(x["region_dna_sequence"].reverse_complement()), axis=1)
    data_df["enhancer_seq"] = data_df.apply(lambda x: str(x["enhancer_seq"]) if x["Start"] < x["TSS"] else str(x["enhancer_seq"].reverse_complement()), axis=1)
    data_df["region_dna_sequence_knockout"] = data_df.apply(lambda x: replace_enhancer_str(x["region_dna_sequence"], x["enhancer_seq"]), axis=1)

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

    output_dir = "/pmglocal/alb2281/get/results/hyena-erythroblast"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    result_col = []
    # iterate over rows of dataframe
    for idx, row in tqdm(data_df.iterrows(), total=len(data_df)):
        if idx % SAVE_EVERY == 0:
            # save results to json
            with open(f"{output_dir}/hyena_erythroblast_benchmark_idx_{idx}.json", "w") as f:
                json.dump(result_col, f)
            result_col = []
        result_col.append(single_inference(row, model))

    if len(result_col) > 0:
        with open(f"{output_dir}/hyena_erythroblast_benchmark_idx_final.json", "w") as f:
            json.dump(result_col, f)
