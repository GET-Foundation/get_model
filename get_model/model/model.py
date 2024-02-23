# simplified GET model
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
# from rotary_embedding_torch import RotaryEmbedding
from get_model.model.position_encoding import CTCFPositionalEncoding, AbsolutePositionalEncoding
from get_model.model.motif import parse_meme_file
from get_model.model.transformer import GETTransformer
from get_model.model.pooling import SplitPool, ATACSplitPool

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
        self, num_motif=637, include_reverse_complement=True, bidirectional_except_ctcf=False):
        super().__init__()
        self.num_motif = num_motif
        self.bidirectional_except_ctcf = bidirectional_except_ctcf
        if include_reverse_complement and self.bidirectional_except_ctcf:
            self.num_motif *= 2
        elif include_reverse_complement:
            self.num_motif *= 2

        motifs = self.load_pwm_as_kernel(include_reverse_complement=include_reverse_complement)
        self.motif = nn.Sequential(
            nn.Conv1d(4, self.num_motif, 29, padding="same", bias=False),
            # nn.BatchNorm1d(num_motif),
            nn.ReLU(),
        )
        assert (
            motifs.shape == self.motif[0].weight.shape
        ), f"Motif prior shape ({motifs.shape}) doesn't match model ({self.motif[0].weight.shape})."
        self.motif[0].weight.data = motifs
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

    def forward(self, x):
        # [B, 200, 1000, 4] --> [B, 200, 1000, 1274]
        B, L, _ = x.shape
        x = x.permute(0, 2, 1)                # (B * N, 4, L)
        x = self.motif(x)
        if self.bidirectional_except_ctcf:
            # get ctcf scanned score for both motif and reverse complement motif. idx of ctcf is 77 and 637+77=714
            ctcf = x[:, 77, :]
            ctcf_rev = x[:, 714, :]
            # combine motif and reverse complement motif for all motifs
            x = x[:, :637, :] + x[:, 637:, :]
            # add ctcf/ctcf_rev score to the end
            x = torch.cat([x, ctcf.unsqueeze(1), ctcf_rev.unsqueeze(1)], dim=1)
        x = x.permute(0, 2, 1)#.reshape(B, L, self.num_motif)     # (B, N, L, 1274)

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
    

class GETPretrain(nn.Module):
    """A GET model for pretraining using mask and prediction."""

    def __init__(
        self,
        num_regions=200,
        num_motif=637,
        motif_dim=639,
        num_res_block=0,
        motif_prior=False,
        embed_dim=768,
        num_layers=12,
        d_model=768,
        nhead=12,
        dropout=0.1,
        output_dim=1,
        pos_emb_components=["CTCF", "Rotary", "Absolute"],
        atac_attention=True,
        flash_attn=False,
        atac_kernel_num=16,
        atac_kernel_size=3,
        joint_kernel_num=16,
        joint_kernel_size=3,
        final_bn=False,
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
        self.atac_kernel_num = atac_kernel_num
        self.atac_kernel_size = atac_kernel_size
        self.joint_kernel_num = joint_kernel_num
        self.joint_kernel_size = joint_kernel_size
        self.motif_scanner = MotifScanner(
            num_motif=num_motif, include_reverse_complement=True,
            bidirectional_except_ctcf=True
        )
        self.atac_attention = ATACSplitPool(pool_method='mean',
                                            atac_kernel_num=atac_kernel_num,
                                            motif_dim=motif_dim,
                                            joint_kernel_num=joint_kernel_num,
                                            atac_kernel_size=atac_kernel_size,
                                            joint_kernel_size=joint_kernel_size,
                                            final_bn=final_bn,
                                            )
        # self.split_pool = SplitPool()
        self.region_embed = RegionEmbed(num_regions, motif_dim+joint_kernel_num, embed_dim)
        self.pos_embed = []
        if "CTCF" in self.pos_emb_components: 
            self.pos_embed.append(CTCFPositionalEncoding(embed_dim))
        # if "Rotary" in self.pos_emb_components:
            # self.pos_embed.append(RotaryEmbedding(embed_dim))
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
            flash_attn=flash_attn,
        )
        # self.head_atac = nn.Conv1d(d_model, 1, 1)
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

    def forward(self, peak_seq, atac, mask, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std):
        """forward function with hooks to return embedding or attention weights."""
        # peak_seq: [B, L, 4]
        # [B, L, 4] --> [B, L, 1274]
        x = self.motif_scanner(peak_seq)
        x = x - motif_mean_std[:,0, :].unsqueeze(1)
        x = x / motif_mean_std[:,1, :].unsqueeze(1)
        x = F.relu(x)
        # [B, L, 1274] --> [B, R, 1274]
        # gloabl pooling inner product with peak
        x_original = self.atac_attention(x, atac, chunk_size, n_peaks, max_n_peaks)
        # x = self.atac_attention(x, atac)
        # x_original = self.split_pool(x, chunk_size, n_peaks, max_n_peaks)

        x = self.region_embed(x_original)
        B, N, C = x_original.shape
        mask_token = self.mask_token.expand(B, N, -1)
        w = mask.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        for pos_emb_component in self.pos_embed:
            if isinstance(pos_emb_component, CTCFPositionalEncoding):
                x = pos_emb_component(x, ctcf_pos)
            else:
                x = pos_emb_component(x)

        x, _ = self.encoder(x, mask=padding_mask) # (N, D)
        x_masked = self.head_mask(x) # (N, Motif)
        # x_masked = x_masked[mask].reshape(B, -1, C)
        # atac = F.softplus(self.head_atac(x.permute(0, 2, 1)).permute(0, 2, 1).squeeze(-1))
        atac = None
        
        return x_masked, atac, x_original

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


class GETFinetune(nn.Module):
    """A GET model for finetuning using classification head."""

    def __init__(
        self,
        num_regions=200,
        num_motif=637,
        motif_dim=639,
        num_res_block=0,
        motif_prior=False,
        embed_dim=768,
        num_layers=12,
        d_model=768,
        nhead=12,
        dropout=0.1,
        output_dim=1,
        pos_emb_components=["CTCF", "Rotary", "Absolute"],
        atac_attention=True,
        flash_attn=False,
        atac_kernel_num=16,
        atac_kernel_size=3,
        joint_kernel_num=16,
        joint_kernel_size=3,
        use_atac=False,
        final_bn=False,
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
        self.atac_kernel_num = atac_kernel_num
        self.atac_kernel_size = atac_kernel_size
        self.joint_kernel_num = joint_kernel_num
        self.joint_kernel_size = joint_kernel_size
        self.motif_scanner = MotifScanner(
            num_motif=num_motif, include_reverse_complement=True,
            bidirectional_except_ctcf=True
        )
        self.atac_attention = ATACSplitPool(
            pool_method='mean',
            atac_kernel_num=atac_kernel_num,
            motif_dim=motif_dim,
            joint_kernel_num=joint_kernel_num,
            atac_kernel_size=atac_kernel_size,
            joint_kernel_size=joint_kernel_size,
            final_bn=final_bn,
        )
        # self.split_pool = SplitPool()
        self.region_embed = RegionEmbed(num_regions, motif_dim+joint_kernel_num, embed_dim)
        self.pos_embed = []
        if "CTCF" in self.pos_emb_components: 
            self.pos_embed.append(CTCFPositionalEncoding(embed_dim))
        # if "Rotary" in self.pos_emb_components:
            # self.pos_embed.append(RotaryEmbedding(embed_dim))
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
            flash_attn=flash_attn,
        )
        # self.head_atac = nn.Conv1d(d_model, 1, 1)
        # self.head_mask = nn.Linear(d_model, output_dim)
        self.head_exp = (
            ExpressionHead(d_model, output_dim, use_atac)
            if output_dim > 0
            else nn.Identity()
        )
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # trunc_normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, peak_seq, atac, mask, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, other_labels):
        """labels_data is (B, R, C), C=2 for expression. R=max_n_peaks"""
        # peak_seq: [B, L, 4]
        # [B, L, 4] --> [B, L, 1274]
        x = self.motif_scanner(peak_seq)
        x = x - motif_mean_std[:,0, :].unsqueeze(1)
        x = x / motif_mean_std[:,1, :].unsqueeze(1)
        x = F.relu(x)
        # [B, L, 1274] --> [B, R, 1274]
        # gloabl pooling inner product with peak
        x_original = self.atac_attention(x, atac, chunk_size, n_peaks, max_n_peaks)
        # x = self.atac_attention(x, atac)
        # x_original = self.split_pool(x, chunk_size, n_peaks, max_n_peaks)

        x = self.region_embed(x_original)
        B, N, C = x_original.shape


        for pos_emb_component in self.pos_embed:
            if isinstance(pos_emb_component, CTCFPositionalEncoding):
                x = pos_emb_component(x, ctcf_pos)
            else:
                x = pos_emb_component(x)

        tss_mask = None # TODO: Set tss_mask to None for now
        x, _ = self.encoder(x, mask=padding_mask)
        # atac = F.softplus(self.head_atac(x.permute(0, 2, 1))).permute(0, 2, 1).squeeze(-1)
        atac = None

        exp = F.softplus(self.head_exp(x, atac))
        return atac, exp

    def reset_head(self, output_dim):
        self.output_dim = output_dim
        self.head_exp = (
            nn.Linear(self.embed_dim, output_dim) if output_dim > 0 else nn.Identity()
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

class GETFinetuneExpATAC(nn.Module):
    """A GET model for finetuning using classification head."""

    def __init__(
        self,
        num_regions=200,
        num_motif=637,
        motif_dim=639,
        num_res_block=0,
        motif_prior=False,
        embed_dim=768,
        num_layers=12,
        d_model=768,
        nhead=12,
        dropout=0.1,
        output_dim=1,
        pos_emb_components=["CTCF", "Rotary", "Absolute"],
        atac_attention=True,
        flash_attn=False,
        atac_kernel_num=16,
        atac_kernel_size=3,
        joint_kernel_num=16,
        joint_kernel_size=3,
        use_atac=False,
        final_bn=False,
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
        self.atac_kernel_num = atac_kernel_num
        self.atac_kernel_size = atac_kernel_size
        self.joint_kernel_num = joint_kernel_num
        self.joint_kernel_size = joint_kernel_size
        self.motif_scanner = MotifScanner(
            num_motif=num_motif, include_reverse_complement=True,
            bidirectional_except_ctcf=True
        )
        self.atac_attention = ATACSplitPool(
            pool_method='mean',
            atac_kernel_num=atac_kernel_num,
            motif_dim=motif_dim,
            joint_kernel_num=joint_kernel_num,
            atac_kernel_size=atac_kernel_size,
            joint_kernel_size=joint_kernel_size,
            final_bn=final_bn,
        )
        # self.split_pool = SplitPool()
        self.region_embed = RegionEmbed(num_regions, motif_dim+joint_kernel_num, embed_dim)
        self.pos_embed = []
        if "CTCF" in self.pos_emb_components: 
            self.pos_embed.append(CTCFPositionalEncoding(embed_dim))
        # if "Rotary" in self.pos_emb_components:
            # self.pos_embed.append(RotaryEmbedding(embed_dim))
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
            flash_attn=flash_attn,
        )
        self.head_atac = ATACHead(motif_dim+joint_kernel_num, d_model, 1)
        # self.head_mask = nn.Linear(d_model, output_dim)
        self.head_exp = (
            ExpressionHead(d_model, output_dim, use_atac)
            if output_dim > 0
            else nn.Identity()
        )
        self.head_confidence = (
            ExpressionHead(d_model, 50, use_atac)
            if output_dim > 0
            else nn.Identity()
        )
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # trunc_normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, peak_seq, atac, mask, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, other_labels):
        """labels_data is (B, R, C), C=2 for expression. R=max_n_peaks"""
        # peak_seq: [B, L, 4]
        # [B, L, 4] --> [B, L, 1274]
        x = self.motif_scanner(peak_seq)
        x = x - motif_mean_std[:,0, :].unsqueeze(1)
        x = x / motif_mean_std[:,1, :].unsqueeze(1)
        x = F.relu(x)
        # [B, L, 1274] --> [B, R, 1274]
        # gloabl pooling inner product with peak
        
        x_original = self.atac_attention(x, atac, chunk_size, n_peaks, max_n_peaks)
        # x = self.atac_attention(x, atac)
        # x_original = self.split_pool(x         , chunk_size, n_peaks, max_n_peaks)
        tss_mask = other_labels[:,:, 1]
        # x_original = torch.cat([x_original, atpm], dim=-1)
        atpm = F.softplus(self.head_atac(x_original))
        x = self.region_embed(x_original)

        B, N, C = x_original.shape


        for pos_emb_component in self.pos_embed:
            if isinstance(pos_emb_component, CTCFPositionalEncoding):
                x = pos_emb_component(x, ctcf_pos)
            else:
                x = pos_emb_component(x)

        x, _ = self.encoder(x, mask=padding_mask)
        exp = F.softplus(self.head_exp(x, None))
        confidence = F.softplus(self.head_confidence(x, None))
        return atpm, exp, confidence

    def reset_head(self, output_dim):
        self.output_dim = output_dim
        self.head_exp = (
            nn.Linear(self.embed_dim, output_dim) if output_dim > 0 else nn.Identity()
        )
        self.head_atac = ATACHead(self.embed_dim, self.d_model, 1)
        self.head_confidence = ExpressionHead(self.embed_dim, 50, False)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

class GETFinetuneATAC(nn.Module):
    """A GET model for finetuning using classification head."""

    def __init__(
        self,
        num_regions=200,
        num_motif=637,
        motif_dim=639,
        num_res_block=0,
        motif_prior=False,
        embed_dim=768,
        num_layers=12,
        d_model=768,
        nhead=12,
        dropout=0.1,
        output_dim=1,
        pos_emb_components=["CTCF", "Rotary", "Absolute"],
        atac_attention=True,
        flash_attn=False,
        atac_kernel_num=16,
        atac_kernel_size=3,
        joint_kernel_num=16,
        joint_kernel_size=3,
        use_atac=True,
        final_bn=False,
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
        self.atac_kernel_num = atac_kernel_num
        self.atac_kernel_size = atac_kernel_size
        self.joint_kernel_num = joint_kernel_num
        self.joint_kernel_size = joint_kernel_size
        self.motif_scanner = MotifScanner(
            num_motif=num_motif, include_reverse_complement=True,
            bidirectional_except_ctcf=True
        )
        self.atac_attention = ATACSplitPool(
            pool_method='mean',
            atac_kernel_num=atac_kernel_num,
            motif_dim=motif_dim,
            joint_kernel_num=joint_kernel_num,
            atac_kernel_size=atac_kernel_size,
            joint_kernel_size=joint_kernel_size,
            final_bn=final_bn,
        )
        # self.split_pool = SplitPool()
        self.region_embed = RegionEmbed(num_regions, motif_dim+joint_kernel_num, embed_dim)
        self.pos_embed = []
        if "CTCF" in self.pos_emb_components: 
            self.pos_embed.append(CTCFPositionalEncoding(embed_dim))
        # if "Rotary" in self.pos_emb_components:
            # self.pos_embed.append(RotaryEmbedding(embed_dim))
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
            flash_attn=flash_attn,
        )
        self.head_atac = ATACHead(d_model, 1)
        # self.head_mask = nn.Linear(d_model, output_dim)
        self.head_exp = (
            ExpressionHead(d_model, output_dim, use_atac)
            if output_dim > 0
            else nn.Identity()
        )
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # trunc_normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, peak_seq, atac, mask, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std):
        """labels_data is (B, R, C), C=2 for expression. R=max_n_peaks"""
        # peak_seq: [B, L, 4]
        # [B, L, 4] --> [B, L, 1274]
        x = self.motif_scanner(peak_seq)
        x = x - motif_mean_std[:,0, :].unsqueeze(1)
        x = x / motif_mean_std[:,1, :].unsqueeze(1)
        x = F.relu(x)
        # [B, L, 1274] --> [B, R, 1274]
        # gloabl pooling inner product with peak
        x_original = self.atac_attention(x, atac, chunk_size, n_peaks, max_n_peaks)
        # x = self.atac_attention(x, atac)
        # x_original = self.split_pool(x, chunk_size, n_peaks, max_n_peaks)

        x = self.region_embed(x_original)
        B, N, C = x_original.shape


        for pos_emb_component in self.pos_embed:
            if isinstance(pos_emb_component, CTCFPositionalEncoding):
                x = pos_emb_component(x, ctcf_pos)
            else:
                x = pos_emb_component(x)

        tss_mask = None # TODO: Set tss_mask to None for now
        x, _ = self.encoder(x, mask=padding_mask)
        atac = F.softplus(self.head_atac(x.permute(0, 2, 1))).permute(0, 2, 1).squeeze(-1)

        exp = F.softplus(self.head_exp(x, atac))
        return atac, exp

    def reset_head(self, output_dim):
        self.output_dim = output_dim
        self.head_exp = (
            nn.Linear(self.embed_dim, output_dim) if output_dim > 0 else nn.Identity()
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}


class GETFinetuneExpATACFromSequence(nn.Module):
    """A GET model for finetuning using classification head."""

    def __init__(
        self,
        num_regions=200,
        num_motif=637,
        motif_dim=639,
        num_res_block=0,
        motif_prior=False,
        embed_dim=768,
        num_layers=12,
        d_model=768,
        nhead=12,
        dropout=0.1,
        output_dim=1,
        pos_emb_components=["CTCF", "Rotary", "Absolute"],
        flash_attn=False,
        use_atac=False,
    ):
        super().__init__()
        self.num_regions = num_regions
        self.num_motif = num_motif
        self.num_res_block = num_res_block
        self.motif_prior = motif_prior
        self.embed_dim = embed_dim
        self.motif_dim = motif_dim
        self.num_layers = num_layers
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.output_dim = output_dim
        self.pos_emb_components = pos_emb_components
        self.motif_scanner = MotifScanner(
            num_motif=num_motif, include_reverse_complement=True,
            bidirectional_except_ctcf=True
        )
        self.atac_attention = SplitPool(
            pool_method='mean',
        )
        # self.split_pool = SplitPool()
        self.region_embed = RegionEmbed(num_regions, motif_dim+joint_kernel_num, embed_dim)
        self.pos_embed = []
        if "CTCF" in self.pos_emb_components: 
            self.pos_embed.append(CTCFPositionalEncoding(embed_dim))
        # if "Rotary" in self.pos_emb_components:
            # self.pos_embed.append(RotaryEmbedding(embed_dim))
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
            flash_attn=flash_attn,
        )
        self.head_atac = ATACHead(motif_dim, d_model, 1)
        # self.head_mask = nn.Linear(d_model, output_dim)
        self.head_exp = (
            ExpressionHead(d_model, output_dim, use_atac)
            if output_dim > 0
            else nn.Identity()
        )
        self.head_confidence = (
            ExpressionHead(d_model, 50, use_atac)
            if output_dim > 0
            else nn.Identity()
        )
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # trunc_normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, peak_seq, atac, mask, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, other_labels):
        """labels_data is (B, R, C), C=2 for expression. R=max_n_peaks"""
        # peak_seq: [B, L, 4]
        # [B, L, 4] --> [B, L, 1274]
        x = self.motif_scanner(peak_seq)
        x = x - motif_mean_std[:,0, :].unsqueeze(1)
        x = x / motif_mean_std[:,1, :].unsqueeze(1)
        x = F.relu(x)
        # [B, L, 1274] --> [B, R, 1274]
        # gloabl pooling inner product with peak
        x_original = self.atac_attention(x, chunk_size, n_peaks, max_n_peaks)
        # x = self.atac_attention(x, atac)
        atpm = F.softplus(self.head_atac(x_original))

        # x_original = self.split_pool(x         , chunk_size, n_peaks, max_n_peaks)
        tss_mask = other_labels[:,:, 1]
        # x_original = torch.cat([x_original, atpm], dim=-1)
        
        x = self.region_embed(x_original)

        B, N, C = x_original.shape


        for pos_emb_component in self.pos_embed:
            if isinstance(pos_emb_component, CTCFPositionalEncoding):
                x = pos_emb_component(x, ctcf_pos)
            else:
                x = pos_emb_component(x)

        x, _ = self.encoder(x, mask=padding_mask)
        exp = F.softplus(self.head_exp(x, None))
        confidence = F.softplus(self.head_confidence(x, None))
        return atpm, exp, confidence

    def reset_head(self, output_dim):
        self.output_dim = output_dim
        self.head_exp = (
            nn.Linear(self.embed_dim, output_dim) if output_dim > 0 else nn.Identity()
        )
        self.head_atac = ATACHead(self.motif_dim, self.d_model, 1)
        self.head_confidence = ExpressionHead(self.embed_dim, 50, False)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}



@register_model
def get_pretrain_motif_base(pretrained=False, **kwargs):
    model = GETPretrain(
        num_regions=kwargs["num_region_per_sample"],
        num_motif=kwargs["num_motif"],
        motif_dim=kwargs["motif_dim"],
        num_res_block=0,
        motif_prior=False,
        embed_dim=768,
        num_layers=12,
        d_model=768,
        nhead=12,
        dropout=0.1,
        output_dim=kwargs["output_dim"],
        pos_emb_components=[],
        flash_attn=kwargs["flash_attn"],
        atac_kernel_num=161,
        joint_kernel_num=161,
        final_bn=kwargs["final_bn"],
    )
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def get_finetune_motif(pretrained=False, **kwargs):
    model = GETFinetune(
        num_regions=kwargs["num_region_per_sample"],
        num_motif=kwargs["num_motif"],
        motif_dim=kwargs["motif_dim"],
        num_res_block=0,
        motif_prior=False,
        embed_dim=768,
        num_layers=12,
        d_model=768,
        nhead=12,
        dropout=0.1,
        output_dim=kwargs["output_dim"],
        pos_emb_components=[],
        flash_attn=kwargs["flash_attn"],
        atac_kernel_num=161,
        joint_kernel_num=161,
        final_bn=kwargs["final_bn"],
    )
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def get_finetune_motif_with_atac(pretrained=False, **kwargs):
    model = GETFinetuneExpATAC(
        num_regions=kwargs["num_region_per_sample"],
        num_motif=kwargs["num_motif"],
        motif_dim=kwargs["motif_dim"],
        num_res_block=0,
        motif_prior=False,
        embed_dim=768,
        num_layers=12,
        d_model=768,
        nhead=12,
        dropout=0.1,
        output_dim=kwargs["output_dim"],
        pos_emb_components=[],
        flash_attn=kwargs["flash_attn"],
        atac_kernel_num=161,
        joint_kernel_num=161,
        final_bn=kwargs["final_bn"],
    )
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def get_finetune_motif_with_atac_from_sequence(pretrained=False, **kwargs):
    model = GETFinetuneExpATACFromSequence(
        num_regions=kwargs["num_region_per_sample"],
        num_motif=kwargs["num_motif"],
        motif_dim=kwargs["motif_dim"],
        num_res_block=0,
        motif_prior=False,
        embed_dim=768,
        num_layers=12,
        d_model=768,
        nhead=12,
        dropout=0.1,
        output_dim=kwargs["output_dim"],
        pos_emb_components=[],
        flash_attn=kwargs["flash_attn"],
    )
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model
