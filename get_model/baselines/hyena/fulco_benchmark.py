import json
import os
import subprocess
import torch

from transformers import PreTrainedModel
import re
from standalone_hyenadna import HyenaDNAModel
from standalone_hyenadna import CharacterTokenizer
from transformers import AutoModelForCausalLM

from atac_rna_data_processing.io.region import Genome
import os
import pandas as pd
import torch 
from tqdm import tqdm
import numpy as np
from Bio.Seq import reverse_complement, Seq


SEQUENCE_LENGTH = 1_000_000
WINDOW_SIZE = 2000


# helper 1
def inject_substring(orig_str):
    """Hack to handle matching keys between models trained with and without
    gradient checkpointing."""

    # modify for mixer keys
    pattern = r"\.mixer"
    injection = ".mixer.layer"

    modified_string = re.sub(pattern, injection, orig_str)

    # modify for mlp keys
    pattern = r"\.mlp"
    injection = ".mlp.layer"

    modified_string = re.sub(pattern, injection, modified_string)

    return modified_string

# helper 2
def load_weights(scratch_dict, pretrained_dict, checkpointing=False):
    """Loads pretrained (backbone only) weights into the scratch state dict."""

    # loop thru state dict of scratch
    # find the corresponding weights in the loaded model, and set it

    # need to do some state dict "surgery"
    for key, value in scratch_dict.items():
        if 'backbone' in key:
            # the state dicts differ by one prefix, '.model', so we add that
            key_loaded = 'model.' + key
            # breakpoint()
            # need to add an extra ".layer" in key
            if checkpointing:
                key_loaded = inject_substring(key_loaded)
            try:
                scratch_dict[key] = pretrained_dict[key_loaded]
            except:
                raise Exception('key mismatch in the state dicts!')

    # scratch_dict has been updated
    return scratch_dict


def inference_batch(data_df, running_file):
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

    # create tokenizer
    tokenizer = CharacterTokenizer(
        characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
        model_max_length=max_length + 2,  # to account for special tokens, like EOS
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side='left', # since HyenaDNA is causal, we pad on the left
    )

    # prep model and forward
    model.to(device)
    model.eval()

    with open(running_file, "w") as output_f:
        with torch.inference_mode():
            #### Single embedding example ####
            for idx, row in tqdm(data_df.iterrows(), total=len(data_df)):
                if not row["contains_enhancer"]:
                    row["region_score"] = np.nan
                    row["knockout_score"] = np.nan
                    output_row = row[["orig_idx", "chrom", "chromStart", "chromEnd", "region_score", "knockout_score"]]
                    output_f.write("\t".join(map(str, output_row.values)) + "\n")
                    output_f.flush()
                else:
                    row["attention_start"] = row["startTSS"] - WINDOW_SIZE // 2
                    row["attention_end"] = row["startTSS"] + WINDOW_SIZE // 2

                    start_idx = row["attention_start"] - row["region_start"]
                    end_idx = row["attention_end"] - row["region_start"]

                    input_ids = np.array(tokenizer(row["region_dna_sequence"])["input_ids"])
                    region_token_seq = torch.LongTensor(input_ids).unsqueeze(0)  # unsqueeze for batch dim
                    region_tok_seq = region_token_seq.to(device)
                    region_logits = model(region_tok_seq).logits.cpu().detach().numpy()
                    region_logits = region_logits[0, np.arange(start_idx, end_idx), input_ids]  # grab logits corresponding to sequence
                    row["region_score"] = np.sum(region_logits)

                    knockout_input_ids = np.array(tokenizer(row["region_dna_sequence_knockout"])["input_ids"])
                    knockout_token_seq = torch.LongTensor(knockout_input_ids).unsqueeze(0)  # unsqueeze for batch dim
                    knockout_tok_seq = knockout_token_seq.to(device)
                    knockout_logits = model(knockout_tok_seq).logits.cpu().detach().numpy()
                    knockout_logits = knockout_logits[0, np.arange(start_idx, end_idx), knockout_input_ids]  # grab logits corresponding to sequence
                    row["knockout_score"] = np.sum(knockout_logits)

                    output_row = row[["orig_idx", "chrom", "chromStart", "chromEnd", "region_score", "knockout_score"]]
                    output_f.write("\t".join(map(str, output_row.values)) + "\n")
                    output_f.flush()

    return


def replace_enhancer_str(region_str, enhancer_str):
    knockout_str = region_str.replace(enhancer_str, "N" * len(enhancer_str))
    return knockout_str


if __name__=="__main__":
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
    fulco_df["region_dna_sequence"] = fulco_df.apply(lambda x: genome.get_sequence(x["chrom"], x["region_start"], x["region_end"]), axis=1)
    # check if enhancer string in region string
    fulco_df["contains_enhancer"] = fulco_df.apply(lambda x: x["enhancer_seq"] in x["region_dna_sequence"], axis=1)

    fulco_df["region_dna_sequence"] = fulco_df["region_dna_sequence"].apply(lambda x: Seq(x))
    fulco_df["enhancer_seq"] = fulco_df["enhancer_seq"].apply(lambda x: Seq(x))
    # if enhancer string is to the right of the TSS, take the reverse complement of the region and enhancer string (causal Hyena)   
    fulco_df["region_dna_sequence"] = fulco_df.apply(lambda x: str(x["region_dna_sequence"]) if x["chromStart"] < x["startTSS"] else str(x["region_dna_sequence"].reverse_complement()), axis=1)
    fulco_df["enhancer_seq"] = fulco_df.apply(lambda x: str(x["enhancer_seq"]) if x["chromStart"] < x["startTSS"] else str(x["enhancer_seq"].reverse_complement()), axis=1)

    fulco_df["region_dna_sequence_knockout"] = fulco_df.apply(lambda x: replace_enhancer_str(x["region_dna_sequence"], x["enhancer_seq"]), axis=1)
    running_file = "/pmglocal/alb2281/get/results/hyena-1m/hyena_dna_1m_context_fulco_scores_progress.tsv"
    region_scores, enhancer_scores = inference_batch(fulco_df, running_file)
