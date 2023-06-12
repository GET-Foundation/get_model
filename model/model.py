# simplified GET model
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
# from rotary_embedding_torch import RotaryEmbedding
from model.position_encoding import CTCFPositionalEncoding, AbsolutePositionalEncoding
from model.motif import parse_meme_file
from model.transformer import GETTransformer

class SequenceEncoder(nn.Module):
    """A sequence encoder based on Conv1D.
    Input: one-hot encoding of DNA sequences of all regions in batch (BATCH_SIZE, NUM_REGION, SEQ_LEN, 4)
    Output: embedding of batch (BATCH_SIZE, NUM_REGION, EMBED_DIM)
    Architecture:
    Conv1D(4, 32, 3, padding='valid', activation='relu')
    """

    def __init__(
        self, num_region, num_motif=637, num_res_block=3, motif_prior=False
    ):
        super().__init__()
        self.num_region = num_region
        self.num_motif = num_motif
        self.num_res_block = num_res_block
        if motif_prior:
            motifs = self.load_pwm_as_kernel()
            self.motif = nn.Sequential(
                nn.Conv1d(4, num_motif, 29, padding="same"),
                nn.BatchNorm1d(num_motif),
                nn.ReLU(),
            )
            assert (
                motifs.shape == self.motif[0].weight.shape
            ), f"Motif prior shape ({motifs.shape}) doesn't match model ({self.motif[0].weight.shape})."
            self.motif[0].weight.data = motifs.cuda()
            self.motif[0].weight.requires_grad = False

        else:
            self.motif = nn.Sequential(
                nn.Conv1d(4, num_motif, 29, padding="same"),
                nn.BatchNorm1d(num_motif),
                nn.ReLU(),
            )
        if num_res_block > 0:
            self.res_blocks = self.get_residue_block()

    def load_pwm_as_kernel(
        self,
        pwm_path="https://resources.altius.org/~jvierstra/projects/motif-clustering-v2.1beta/consensus_pwms.meme",
    ):
        # download pwm to local
        if not os.path.exists("consensus_pwms.meme"):
            os.system(f"wget {pwm_path}")
        # load pwm
        motifs = parse_meme_file("consensus_pwms.meme")
        return torch.tensor(motifs).permute(0, 2, 1).float()

    def get_residue_block(self):
        res_blocks = []
        for i in range(self.num_res_block):
            res_blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        self.num_motif,
                        self.num_motif,
                        3,
                        padding="same",
                        dilation=2**i,
                    ),
                    nn.BatchNorm1d(self.num_motif),
                    nn.ReLU(),
                )
            )
        return nn.ModuleList(res_blocks)

    def forward(self, x):
        B, N, L, _ = x.shape
        x = x.reshape(B * N, L, 4).permute(
            0, 2, 1
        )  # (BATCH_SIZE * NUM_REGION, 4, SEQ_LEN)
        x = self.motif(x)
        # residue block
        if hasattr(self, "res_blocks"):
            for res_block in self.res_blocks:
                x = x + res_block(x)
        # global average pooling across DNA sequence
        x_mean = x.mean(dim=2)  # (BATCH_SIZE * NUM_REGION, NUM_MOTIF)
        x_mean = x_mean.reshape(-1, self.num_region, self.num_motif)
        return x_mean

    # make sure cuda is used
    def cuda(self, device=None):
        self.motif = self.motif.cuda()
        if hasattr(self, "res_blocks"):
            self.res_blocks = self.res_blocks.cuda()
        return self._apply(lambda t: t.cuda(device))


class RegionEmbed(nn.Module):
    """A simple region embedding transforming motif features to region embeddings.
    Using Conv1D to enforce Linear transformation region-wise.
    """

    def __init__(self, num_regions, num_features, embed_dim):
        super().__init__()
        self.num_region = num_regions
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.embed = nn.Conv1d(num_features, embed_dim, 1)

    def forward(self, x, **kwargs):
        x = x.permute(0, 2, 1)  # (BATCH_SIZE, NUM_MOTIF, NUM_REGION)
        x = self.embed(x)
        x = x.permute(0, 2, 1)  # (BATCH_SIZE, NUM_REGION, EMBED_DIM)
        return x


class GETPretrain(nn.Module):
    """A GET model for pretraining using mask and prediction."""

    def __init__(
        self,
        num_regions=200,
        num_motif=637,
        num_res_block=0,
        motif_prior=False,
        embed_dim=768,
        num_layers=12,
        d_model=768,
        nhead=12,
        dropout=0.1,
        output_dim=1,
        pos_emb_components=["CTCF", "Rotary", "Absolute"],
    ):
        super().__init__()
        self.num_regions = num_regions
        self.num_motif = num_motif
        self.num_res_block = num_res_block
        self.motif_prior = motif_prior
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.output_dim = output_dim
        self.pos_emb_components = pos_emb_components
        self.motif = SequenceEncoder(
            num_regions, num_motif, num_res_block, motif_prior
        )
        self.region_embed = RegionEmbed(num_regions, num_motif, embed_dim)
        self.pos_embed = []
        if "CTCF" in self.pos_emb_components: 
            self.pos_embed.append(CTCFPositionalEncoding(embed_dim))
        if "Rotary" in self.pos_emb_components:
            self.pos_embed.append(RotaryEmbedding(embed_dim))
        if "Absolute" in self.pos_emb_components:
            self.pos_embed.append(AbsolutePositionalEncoding(embed_dim))
        self.pos_embed = nn.ModuleList(self.pos_embed)
        self.encoder = GETTransformer(
            d_model,
            nhead,
            num_layers,
            drop_path_rate=dropout,
            drop_rate=dropout,
            attn_drop_rate=dropout,
            use_mean_pooling=False,
        )
        self.head_atac = nn.Conv1d(d_model, 1, 1)
        self.head_mask = nn.Linear(d_model, output_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask, ctcf_pos):
        """forward function with hooks to return embedding or attention weights."""
        x = self.motif(
            x
        )  # TODO ignore the nucleotide level output x for now, keeping only the region level output
        x = self.region_embed(x)

        B, N, C = x.shape
        mask_token = self.mask_token.expand(B, N, -1)
        w = mask.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        for pos_emb_component in self.pos_embed:
            if isinstance(pos_emb_component, CTCFPositionalEncoding):
                x = pos_emb_component(x, ctcf_pos)
            else:
                x = pos_emb_component(x)

        x, _ = self.encoder(x, mask=mask) # (N, D)
        x_masked = self.head_mask(x) # (N, Motif)
        x_masked = x_masked[mask].reshape(B, -1, C)
        atac = F.softplus(self.head_atac(x.permute(0, 2, 1)).permute(0, 2, 1))

        return x_masked, atac

    def reset_head(self, output_dim, global_pool=""):
        self.output_dim = output_dim
        self.head = (
            nn.Linear(self.embed_dim, output_dim) if output_dim > 0 else nn.Identity()
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}


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


class GETFinetune(GETPretrain):
    """A GET model for finetuning using classification head."""

    def __init__(
        self,
        num_regions=200,
        num_motif=637,
        num_res_block=0,
        motif_prior=False,
        embed_dim=768,
        num_layers=8,
        d_model=768,
        nhead=8,
        dropout=0.1,
        output_dim=2,
        pos_emb_components=["CTCF", "Rotary", "Absolute"],
        use_atac=True,
    ):
        super().__init__(
            num_regions,
            num_motif,
            num_res_block,
            motif_prior,
            embed_dim,
            num_layers,
            d_model,
            nhead,
            dropout,
            output_dim,
            pos_emb_components,
        )
        self.head = nn.Conv1d(self.embed_dim, 1, 1)
        self.head_exp = (
            ExpressionHead(embed_dim, output_dim, use_atac)
            if output_dim > 0
            else nn.Identity()
        )

    def forward(self, x, tss_mask, ctcf_pos):
        """forward function with hooks to return embedding or attention weights."""
        x = self.motif(
            x
        )  # TODO ignore the nucleotide level output x for now, keeping only the region level output
        x = self.region_embed(x)
        # B, N, C = x.shape

        for pos_emb_component in self.pos_embed:
            if isinstance(pos_emb_component, CTCFPositionalEncoding):
                x = pos_emb_component(x, ctcf_pos)
            else:
                x = pos_emb_component(x)

        x, _ = self.encoder(x, mask=tss_mask)
        atac = F.softplus(self.head(x.permute(0, 2, 1)).permute(0, 2, 1))
        exp = F.softplus(self.head_exp(x, atac))

        return x, atac, exp

    def reset_head(self, output_dim):
        self.output_dim = output_dim
        self.head_exp = (
            nn.Linear(self.embed_dim, output_dim) if output_dim > 0 else nn.Identity()
        )


@register_model
def get_pretrain_motif(pretrained=False, **kwargs):
    model = GETPretrain(
        num_regions=200,
        num_motif=637,
        num_res_block=1,
        motif_prior=True,
        embed_dim=768,
        num_layers=8,
        d_model=768,
        nhead=8,
        dropout=0.1,
        output_dim=282,
        pos_emb_components=["CTCF"],
    )
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def get_finetune_motif(pretrained=False, **kwargs):
    model = GETFinetune(
        num_regions=200,
        num_motif=637,
        num_res_block=3,
        motif_prior=True,
        embed_dim=768,
        num_layers=8,
        nhead=8,
        dropout=0.1,
        output_dim=2,
        pos_emb_components=["CTCF"],
    )
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

# %%
