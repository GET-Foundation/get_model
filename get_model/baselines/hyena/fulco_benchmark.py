import json
import os
import subprocess
import torch

from transformers import PreTrainedModel
import re
from standalone_hyenadna import HyenaDNAModel
from standalone_hyenadna import CharacterTokenizer

from atac_rna_data_processing.io.region import Genome
import os
import pandas as pd
import torch 
from tqdm import tqdm

SEQUENCE_LENGTH = 1_000_000


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

class HyenaDNAPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    base_model_prefix = "hyenadna"

    def __init__(self, config):
        pass

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    @classmethod
    def from_pretrained(cls,
                        path,
                        model_name,
                        download=False,
                        config=None,
                        device='cpu',
                        use_head=False,
                        n_classes=2,
                      ):
        # first check if it is a local path
        pretrained_model_name_or_path = os.path.join(path, model_name)
        if os.path.isdir(pretrained_model_name_or_path) and download == False:
            if config is None:
                config = json.load(open(os.path.join(pretrained_model_name_or_path, 'config.json')))
        else:
            hf_url = f'https://huggingface.co/LongSafari/{model_name}'

            subprocess.run(f'rm -rf {pretrained_model_name_or_path}', shell=True)
            command = f'mkdir -p {path} && cd {path} && git lfs install && git clone {hf_url}'
            subprocess.run(command, shell=True)

            if config is None:
                config = json.load(open(os.path.join(pretrained_model_name_or_path, 'config.json')))

        scratch_model = HyenaDNAModel(**config, use_head=use_head, n_classes=n_classes)  # the new model format
        loaded_ckpt = torch.load(
            os.path.join(pretrained_model_name_or_path, 'weights.ckpt'),
            map_location=torch.device(device)
        )

        # need to load weights slightly different if using gradient checkpointing
        if config.get("checkpoint_mixer", False):
            checkpointing = config["checkpoint_mixer"] == True or config["checkpoint_mixer"] == True
        else:
            checkpointing = False

        # grab state dict from both and load weights
        state_dict = load_weights(scratch_model.state_dict(), loaded_ckpt['state_dict'], checkpointing=checkpointing)

        # scratch model has now been updated
        scratch_model.load_state_dict(state_dict)
        print("Loaded pretrained weights ok!")
        return scratch_model


"""# Inference (450k to 1M tokens)!

If all you're interested in is getting embeddings on long DNA sequences
(inference), then we can do that right here in Colab!


*   We provide an example how to load the weights from Huggingface.
*   On the free tier, which uses a
T4 GPU w/16GB of memory, we can process 450k tokens / nucleotides.
*   For processing 1M tokens, you'll need an A100, which Colab offers as a paid tier.
*   (Don't forget to run the entire notebook above too)

--

To pretrain or fine-tune the 1M long sequence model (8 layers, d_model=256),
you'll need 8 A100s 80GB, and all that code is in the main repo!
"""

def inference_batch(data_df, running_file):

    '''
    this selects which backbone to use, and grabs weights/ config from HF
    4 options:
      'hyenadna-tiny-1k-seqlen'   # fine-tune on colab ok
      'hyenadna-small-32k-seqlen'
      'hyenadna-medium-160k-seqlen'  # inference only on colab
      'hyenadna-medium-450k-seqlen'  # inference only on colab
      'hyenadna-large-1m-seqlen'  # inference only on colab
    '''

    # you only need to select which model to use here, we'll do the rest!
    pretrained_model_name = 'hyenadna-large-1m-seqlen'

    max_lengths = {
        'hyenadna-tiny-1k-seqlen': 1024,
        'hyenadna-small-32k-seqlen': 32768,
        'hyenadna-medium-160k-seqlen': 160000,
        'hyenadna-medium-450k-seqlen': 450000,  # T4 up to here
        'hyenadna-large-1m-seqlen': 1_000_000,  # only A100 (paid tier)
    }

    max_length = max_lengths[pretrained_model_name]  # auto selects

    # data settings:
    use_padding = True
    rc_aug = False  # reverse complement augmentation
    add_eos = False  # add end of sentence token

    # we need these for the decoder head, if using
    use_head = False
    n_classes = 2  # not used for embeddings only

    # you can override with your own backbone config here if you want,
    # otherwise we'll load the HF one in None
    backbone_cfg = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # instantiate the model (pretrained here)
    if pretrained_model_name in ['hyenadna-tiny-1k-seqlen',
                                 'hyenadna-small-32k-seqlen',
                                 'hyenadna-medium-160k-seqlen',
                                 'hyenadna-medium-450k-seqlen',
                                 'hyenadna-large-1m-seqlen']:
        # use the pretrained Huggingface wrapper instead
        model = HyenaDNAPreTrainedModel.from_pretrained(
            './checkpoints',
            pretrained_model_name,
            download=True,
            config=backbone_cfg,
            device=device,
            use_head=use_head,
            n_classes=n_classes,
        )

    # from scratch
    elif pretrained_model_name is None:
        model = HyenaDNAModel(**backbone_cfg, use_head=use_head, n_classes=n_classes)

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

    region_scores = []
    enhancer_scores = []

    with open(running_file, "w") as output_f:
        with torch.inference_mode():
            #### Single embedding example ####
            for idx, row in tqdm(data_df.iterrows(), total=len(data_df)):
                region_seq = row["region_seq"]
                enhancer_seq = row["enhancer_seq"]

                region_tok_seq = tokenizer(region_seq)
                region_tok_seq = region_tok_seq["input_ids"]  # grab ids
                region_tok_seq = torch.LongTensor(region_tok_seq).unsqueeze(0)  # unsqueeze for batch dim
                region_tok_seq = region_tok_seq.to(device)
                region_embeddings = model(region_tok_seq)
                region_mean_emb = region_embeddings.mean(axis=1)
                region_norm = torch.norm(region_mean_emb)

                enhancer_tok_seq = tokenizer(enhancer_seq)
                enhancer_tok_seq = enhancer_tok_seq["input_ids"]  # grab ids
                enhancer_tok_seq = torch.LongTensor(enhancer_tok_seq).unsqueeze(0)  # unsqueeze for batch dim
                enhancer_tok_seq = enhancer_tok_seq.to(device)
                enhancer_embeddings = model(enhancer_tok_seq)
                enhancer_mean_emb = enhancer_embeddings.mean(axis=1)
                enhancer_norm = torch.norm(enhancer_mean_emb)

                enhancer_norm = enhancer_norm.cpu().numpy().item()
                region_norm = region_norm.cpu().numpy().item()

                region_scores.append(region_norm)
                enhancer_scores.append(enhancer_norm)
                row["region_score"] = region_norm
                row["enhancer_score"] = enhancer_norm
                row["fulco_score"] = enhancer_norm / region_norm

                output_row = row[["chrom", "chromStart", "chromEnd", "region_score", "enhancer_score", "fulco_score", "orig_idx"]]
                print(output_row.values)
                output_f.write("\t".join(map(str, output_row.values)) + "\n")
                output_f.flush()

    return region_scores, enhancer_scores


if __name__=="__main__":
    fulco_data = "/pmglocal/alb2281/repos/CRISPR_comparison/resources/crispr_data/EPCrisprBenchmark_ensemble_data_GRCh38.tsv"
    fulco_df = pd.read_csv(fulco_data, sep="\t")
    fulco_df["orig_idx"] = fulco_df.index
    fulco_df = fulco_df[fulco_df["startTSS"].notna()]

    genome_path = os.path.join("/manitou/pmg/users/xf2217/interpret_natac/", "hg38.fa")
    genome = Genome('hg38', genome_path)

    fulco_df["enhancer_seq"] = fulco_df.apply(lambda x: genome.get_sequence(x["chrom"], x["chromStart"], x["chromEnd"]).seq, axis=1)
    fulco_df["region_start"] = fulco_df["startTSS"] - SEQUENCE_LENGTH // 2
    fulco_df["region_end"] = fulco_df["startTSS"] + SEQUENCE_LENGTH // 2
    fulco_df["region_start"] = fulco_df["region_start"].apply(lambda x: int(x))
    fulco_df["region_end"] = fulco_df["region_end"].apply(lambda x: int(x))
    fulco_df["region_seq"] = fulco_df.apply(lambda x: genome.get_sequence(x["chrom"], x["region_start"], x["region_end"]).seq, axis=1)

    running_file = "/pmglocal/alb2281/get/results/hyena-1m/hyena_dna_1m_context_fulco_scores_progress.tsv"
    region_scores, enhancer_scores = inference_batch(fulco_df, running_file)
    fulco_df["region_score"] = region_scores
    fulco_df["enhancer_score"] = enhancer_scores
    fulco_df["fulco_score"] = fulco_df["enhancer_score"] / fulco_df["region_score"]
    fulco_df.to_csv("/pmglocal/alb2281/get/results/hyena-1m/hyena_dna_1m_context_fulco_scores.tsv", sep="\t", index=False)
