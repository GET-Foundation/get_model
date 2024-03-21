# simplified GET model
import os
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
# from rotary_embedding_torch import RotaryEmbedding
from get_model.model.position_encoding import CTCFPositionalEncoding, AbsolutePositionalEncoding
from get_model.model.motif import parse_meme_file
from get_model.model.transformer import GETTransformer
from get_model.model.pooling import SplitPool, ATACSplitPool, ATACSplitPoolMaxNorm, ConvPool
from torch.nn import MSELoss
import torch
from torch.nn import Linear

class RegionEmbed(nn.Module):
    """A simple region embedding transforming motif features to region embeddings.
    Using Conv1D to enforce Linear transformation region-wise.
    """

    def __init__(self, num_regions, num_features, embed_dim):
        super().__init__()
        self.num_region = num_regions
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.embed = nn.Linear(num_features, embed_dim)

    def forward(self, x, **kwargs):
        # x = x.permute(0, 2, 1)  # (BATCH_SIZE, NUM_MOTIF, NUM_REGION)
        x = self.embed(x)
        # x = x.permute(0, 2, 1)  # (BATCH_SIZE, NUM_REGION, EMBED_DIM)
        return x

class TFEncoder(nn.Module):
    """This module is used to encode TF protein information. More specifically,
    each TF among L TFs is represented by a N-by-D matrix (as input), where N is 1 or 2 or N, 
    when N is 1, the vector is directly from ESM CLS token embedding
    when N is 2, the two vectors are from DBD CLS and non-DBD CLS token embedding
    when N is N, the vectors are from pLDDT-segmented CLS token embedding
    the output of this module is a L by D matrix which captures the TF relationship.
    """
    pass


class MotifScanner(nn.Module):
    """A motif encoder based on Conv1D.
    Input: one-hot encoding of DNA sequences of all regions in batch (BATCH_SIZE, NUM_REGION, SEQ_LEN, 4)
    Output: embedding of batch (BATCH_SIZE, NUM_REGION, EMBED_DIM)
    Architecture:
    Conv1D(4, 32, 3, padding='valid', activation='relu')
    """

    def __init__(
        self, num_motif=637, include_reverse_complement=True, bidirectional_except_ctcf=False, motif_prior=True, learnable=False):
        super().__init__()
        self.num_motif = num_motif
        self.bidirectional_except_ctcf = bidirectional_except_ctcf
        self.motif_prior = motif_prior
        self.learnable = learnable
        if include_reverse_complement and self.bidirectional_except_ctcf:
            self.num_motif *= 2
        elif include_reverse_complement:
            self.num_motif *= 2

        motifs = self.load_pwm_as_kernel(include_reverse_complement=include_reverse_complement)
        self.motif = nn.Sequential(
            nn.Conv1d(4, self.num_motif, 29, padding="same", bias=True),
            # nn.BatchNorm1d(num_motif),
            nn.ReLU(),
        )
        if self.motif_prior:
            assert (
                motifs.shape == self.motif[0].weight.shape
            ), f"Motif prior shape ({motifs.shape}) doesn't match model ({self.motif[0].weight.shape})."
            self.motif[0].weight.data = motifs
        if not self.learnable:
            self.motif[0].weight.requires_grad = False
    
    def load_pwm_as_kernel(
        self,
        pwm_path="https://resources.altius.org/~jvierstra/projects/motif-clustering-v2.1beta/consensus_pwms.meme",
        include_reverse_complement=True,
    ):
        # download pwm to local
        if not os.path.exists("consensus_pwms.meme"):
            os.system(f"wget {pwm_path}")
        # load pwm
        motifs = parse_meme_file("consensus_pwms.meme")
        motifs_rev = motifs[:, ::-1, ::-1].copy()
        # construct reverse complement
        motifs = torch.tensor(motifs)
        motifs_rev = torch.tensor(motifs_rev)
        motifs = torch.cat([motifs, motifs_rev], dim=0)
        return motifs.permute(0, 2, 1).float()

    def normalize_motif(self, x, motif_mean_std):
        return (x - motif_mean_std[:, 0, :].unsqueeze(1)) / motif_mean_std[:, 1, :].unsqueeze(1)
    
    def forward(self, x, motif_mean_std=None):
        x = x.permute(0, 2, 1) 
        x = self.motif(x)
        if self.bidirectional_except_ctcf:
            # get ctcf scanned score for both motif and reverse complement motif. idx of ctcf is 77 and 637+77=714
            ctcf = x[:, 77, :]
            ctcf_rev = x[:, 714, :]
            # combine motif and reverse complement motif for all motifs
            x = x[:, :637, :] + x[:, 637:, :]
            # add ctcf/ctcf_rev score to the end
            x = torch.cat([x, ctcf.unsqueeze(1), ctcf_rev.unsqueeze(1)], dim=1)
        x = x.permute(0, 2, 1)
        if motif_mean_std is not None:
            x = self.normalize_motif(x, motif_mean_std)
        x = F.relu(x)
        return x

    # make sure cuda is used
    def cuda(self, device=None):
        self.motif = self.motif.cuda()
        return self._apply(lambda t: t.cuda(device))

class ATACAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, peak_seq, atac):
        return peak_seq * atac.unsqueeze(-1)
        # return torch.einsum("bld,blc->blcd", peak_seq, atac).sum(dim=2)
    


class ExpressionHead(nn.Module):
    """Expression head"""

    def __init__(self, embed_dim, output_dim, use_atac=False):
        super().__init__()
        self.use_atac = use_atac
        if use_atac:
            self.head = nn.Linear(embed_dim + 1, output_dim)
        else:
            self.head = nn.Linear(embed_dim, output_dim)

    def forward(self, x, atac=None):
        if self.use_atac:
            x = torch.cat([x, atac], dim=-1)
        return self.head(x)

class ATACHead(nn.Module):
    """ATAC head"""

    def __init__(self, embed_dim, hidden_dim, output_dim, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
