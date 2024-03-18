# simplified GET model
import os
from collections import OrderedDict
from typing import Optional

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model

from get_model.model.modules import *
from get_model.model.motif import parse_meme_file
from get_model.model.pooling import (ATACSplitPool, ATACSplitPoolMaxNorm,
                                     ConvPool, SplitPool)
# from rotary_embedding_torch import RotaryEmbedding
from get_model.model.position_encoding import (AbsolutePositionalEncoding,
                                               CTCFPositionalEncoding)
from get_model.model.transformer import GETTransformer
from get_model.utils import freeze_layers, load_checkpoint, load_state_dict, remove_keys, rename_keys


class BaseGETModel(nn.Module):
    def __init__(self, cfg):
        super(BaseGETModel, self).__init__()
        self.loss_specification = cfg.loss_specification
        self.stats_specification = cfg.stats_specification
        # Define common model properties and methods
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def reset_head(self, output_dim, global_pool=""):
        self.output_dim = output_dim
        self.head = (
            nn.Linear(self.embed_dim, output_dim) if output_dim > 0 else nn.Identity()
        )

    def data_prep(self, batch):
        # Define the data preparation logic
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        # Define the forward pass of the model
        raise NotImplementedError

    def before_loss(self, *args, **kwargs):
        # Define the training statistics
        raise NotImplementedError
    
    def loss_fn(self):
        # Define the loss function
        raise NotImplementedError

    def test_stats(self, *args, **kwargs):
        # Define the testing statistics
        raise NotImplementedError
    
    def global_test_stats(self, *args, **kwargs):
        # Define the global testing statistics
        raise NotImplementedError
    
class GETPretrainMaxNorm(nn.Module):
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
            bidirectional_except_ctcf=True,
            motif_prior=motif_prior,
            learnable=False,
        )
        self.atac_attention = ATACSplitPoolMaxNorm(pool_method='mean',
                                            atac_kernel_num=atac_kernel_num,
                                            motif_dim=motif_dim,
                                            joint_kernel_num=joint_kernel_num,
                                            atac_kernel_size=atac_kernel_size,
                                            joint_kernel_size=joint_kernel_size,
                                            final_bn=final_bn,
                                            atac_input_norm=True
                                            )
        self.split_pool = SplitPool()
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
        # update running max
        # up to this point, x is the motif scanning result and no learning is involved
        x_region = self.split_pool(x, chunk_size, n_peaks, max_n_peaks)

        x_original = self.atac_attention(x, x_region, atac, chunk_size, n_peaks, max_n_peaks)
        # x = self.atac_attention(x, atac)

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

class GETPretrain(BaseGETModel):
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
            bidirectional_except_ctcf=True,
            motif_prior=motif_prior,
            learnable=False,
        )
        self.atac_attention = ATACSplitPool(pool_method='mean',
                                            atac_kernel_num=atac_kernel_num,
                                            motif_dim=motif_dim,
                                            joint_kernel_num=joint_kernel_num,
                                            atac_kernel_size=atac_kernel_size,
                                            joint_kernel_size=joint_kernel_size,
                                            final_bn=final_bn,
                                            atac_input_norm=True
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

    def data_prep(self, batch):
        """Prepare data for forward pass."""
        device = self.device
        atac = batch['sample_track']
        peak_seq = batch['peak_seq']
        loss_mask = batch['loss_mask']
        padding_mask = batch['padding_mask']
        chunk_size = batch['chunk_size']
        n_peaks = batch['n_peaks']
        max_n_peaks = batch['max_n_peaks']
        motif_mean_std = batch['motif_mean_std']
        
        atac = atac.to(device, non_blocking=True).bfloat16()
        peak_seq = peak_seq.to(device, non_blocking=True).bfloat16()
        motif_mean_std = motif_mean_std.to(device, non_blocking=True).bfloat16()
        loss_mask = loss_mask.to(device, non_blocking=True).bool()
        padding_mask = padding_mask.to(device, non_blocking=True).bool()
        n_peaks = n_peaks.to(device, non_blocking=True)
        return peak_seq, atac, loss_mask, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std

    def forward(self, peak_seq, atac, mask, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std):
        """forward function with hooks to return embedding or attention weights."""
        x = self.motif_scanner(peak_seq, motif_mean_std)
        x_original = self.atac_attention(x, atac, chunk_size, n_peaks, max_n_peaks)
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
        
        return x_masked, x_original
    
    def generate_target(self, target, normalize_target=False):
        # target generation
        with torch.no_grad():
            unnorm_targets = target
            if normalize_target:
                print('normalize target')
                regions_squeeze = unnorm_targets
                regions_norm = (
                    regions_squeeze - regions_squeeze.mean(dim=-2, keepdim=True)
                ) / (
                    regions_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt()
                    + 1e-6
                )
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                regions_embed = regions_norm
            else:
                regions_embed = unnorm_targets
        return regions_embed

    def before_loss(self, batch, output):
        """Prepare data for loss calculation."""
        mask_for_loss = batch['mask']
        output_masked, regions_embed = output
        regions_embed = self.generate_target(regions_embed, normalize_target=False)
        mask_for_loss = mask_for_loss.unsqueeze(-1)
        prediction = output_masked*mask_for_loss
        observation = regions_embed*mask_for_loss
        return prediction, observation

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}


class GETFinetune(BaseGETModel):
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
            bidirectional_except_ctcf=True,
            motif_prior=motif_prior,
            learnable=False,
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
        return atac, exp, None

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
            bidirectional_except_ctcf=True,
            motif_prior=motif_prior,
            learnable=False,
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
            bidirectional_except_ctcf=True,
            motif_prior=motif_prior,
            learnable=False,
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
        return atac, exp, None

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
            bidirectional_except_ctcf=True,
            motif_prior=motif_prior,
            learnable=False,
        )
        self.atac_attention = SplitPool(
            pool_method='mean',
        )
        # self.split_pool = SplitPool()
        self.region_embed = RegionEmbed(num_regions, motif_dim, embed_dim)
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

class GETFinetuneExpATACWithHiC(nn.Module):
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
            bidirectional_except_ctcf=True,
            motif_prior=motif_prior,
            learnable=False,
        )
        self.atac_attention = ATACSplitPool(
            pool_method='mean',
            atac_kernel_num=atac_kernel_num,
            motif_dim=motif_dim,
            joint_kernel_num=joint_kernel_num,
            atac_kernel_size=atac_kernel_size,
            joint_kernel_size=joint_kernel_size,
            final_bn=final_bn,
            binary_atac=True,
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

    def forward(self, peak_seq, atac, mask, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, other_labels, hic_matrix):
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

        x, _ = self.encoder(x, mask=padding_mask, bias=hic_matrix)
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

class GETFinetuneChrombpNet(nn.Module):
    def __init__(
        self,
        num_motif=637,
        motif_dim=639,
        embed_dim=768,
        num_regions=200,
        motif_prior=True,
        learnable_motif=False,
        num_layers=7,
        d_model=768,
        nhead=1,
        dropout=0.1,
        output_dim=1,
        with_bias=False,
        bias_ckpt=None,
    ):
        super().__init__()
        self.num_regions = num_regions
        self.num_motif = num_motif
        self.motif_prior = motif_prior
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.output_dim = output_dim
        self.motif_scanner = MotifScanner(
            num_motif=num_motif, include_reverse_complement=True,
            bidirectional_except_ctcf=True,
            motif_prior=motif_prior,
            learnable=learnable_motif,
        )
        self.atac_attention = ConvPool(
            pool_method='mean',
            n_dil_layers=self.num_layers,
            motif_dim=motif_dim,
            hidden_dim=embed_dim
        )
        self.with_bias = with_bias
        if self.with_bias:
            self.bias_model = GETFinetuneChrombpNetBias(
                num_motif=128,
                motif_dim=128,
                embed_dim=128,
                motif_prior=False,
                learnable_motif=True,
            )
            if bias_ckpt is not None:
                checkpoint = torch.load(bias_ckpt, map_location="cpu")
                self.bias_model.load_state_dict(checkpoint["model"])
                # freeze the bias model
                for param in self.bias_model.parameters():
                    param.requires_grad = False

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, peak_seq, atac, mask, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, other_labels, hic_matrix):
        """labels_data is (B, R, C), C=2 for expression. R=max_n_peaks"""
        # peak_seq: [B, L, 4]
        # [B, L, 4] --> [B, L, 1274]
        x = self.motif_scanner(peak_seq)
        x = x - motif_mean_std[:,0, :].unsqueeze(1)
        x = x / motif_mean_std[:,1, :].unsqueeze(1)
        x = F.relu(x)
        atpm, aprofile = self.atac_attention(x, chunk_size, n_peaks, max_n_peaks)
        if self.with_bias:
            bias_atpm, bias_aprofile = self.bias_model(peak_seq, atac, mask, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, other_labels, hic_matrix)
            atpm = torch.logsumexp(torch.stack([atpm, bias_atpm], dim=0), dim=0)
            # pad bias_aprofile to the same length as aprofile
            diff_length = aprofile.shape[1] - bias_aprofile.shape[1]
            crop_length = diff_length // 2
            bias_aprofile = F.pad(bias_aprofile, (crop_length, diff_length - crop_length), "constant", 0)
            aprofile = aprofile + bias_aprofile
            io = {'atpm_output', atpm, 
                'aprofile_output', aprofile,
                'atpm_target', other_labels[:, :, 0],
                'aprofile_target', atac}
            return io
    
    def loss(self, io, config, metric):
        atpm_loss = F.mse_loss(io['atpm_output'], io['atpm_target'])
        aprofile_loss = F.mse_loss(io['aprofile_output'], io['aprofile_target'])
        atpm_loss_weight = config['atpm_loss_weight'] if 'atpm_loss_weight' in config else 1
        aprofile_loss_weight = config['aprofile_loss_weight'] if 'aprofile_loss_weight' in config else 1
        loss = atpm_loss * atpm_loss_weight + aprofile_loss * aprofile_loss_weight
        metric['atpm_loss'] = atpm_loss
        metric['aprofile_loss'] = aprofile_loss
        metric['loss'] = loss
        return loss, metric
        

    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

class GETFinetuneChrombpNetBias(nn.Module):
    """A GET model for finetuning using classification head."""

    def __init__(
        self,
        num_motif=128,
        motif_dim=128,
        embed_dim=128,
        num_regions=1,
        motif_prior=False,
        learnable_motif=True,
        num_layers=4,
        d_model=128,
        nhead=1,
        dropout=0.1,
        output_dim=1,
    ):
        super().__init__()
        self.num_regions = num_regions
        self.num_motif = num_motif
        self.motif_prior = motif_prior
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.output_dim = output_dim
        self.motif_scanner = MotifScanner(
            num_motif=num_motif, include_reverse_complement=False,
            bidirectional_except_ctcf=False,
            motif_prior=motif_prior,
            learnable=learnable_motif,
        )
        self.atac_attention = ConvPool(
            pool_method='mean',
            n_dil_layers=self.num_layers,
            motif_dim=motif_dim,
            hidden_dim=embed_dim
        )
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, peak_seq, atac, mask, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, other_labels, hic_matrix):
        """labels_data is (B, R, C), C=2 for expression. R=max_n_peaks"""
        # peak_seq: [B, L, 4]
        # [B, L, 4] --> [B, L, 1274]
        x = self.motif_scanner(peak_seq)
        atpm, aprofile = self.atac_attention(x, chunk_size, n_peaks, max_n_peaks)
        return atpm, aprofile

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

class GETFinetuneExpATACFromChrombpNet(nn.Module):
    """A GET model for finetuning using classification head."""

    def __init__(
        self,
        num_regions=200,
        num_motif=637,
        motif_dim=639,
        num_res_block=0,
        motif_prior=True,
        learnable_motif=False,
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
            bidirectional_except_ctcf=True,
            motif_prior=motif_prior,
            learnable=learnable_motif,
        )
        self.atac_attention = ConvPool(
            pool_method='mean',
            n_dil_layers=self.num_layers,
            motif_dim=motif_dim,
            hidden_dim=embed_dim//2
        )
        # self.split_pool = SplitPool()
        self.region_embed = RegionEmbed(num_regions, motif_dim, embed_dim)
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
def get_pretrain_motif_base_maxnorm(pretrained=False, **kwargs):
    model = GETPretrainMaxNorm(
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

@register_model
def get_finetune_motif_with_atac_hic(pretrained=False, **kwargs):
    model = GETFinetuneExpATACWithHiC(
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
def get_finetune_motif_chrombpnet(pretrained=False, **kwargs):
    model = GETFinetuneChrombpNet(
        num_motif=kwargs["num_motif"],
        motif_dim=kwargs["motif_dim"],
        embed_dim=512,
        motif_prior=True,
        learnable_motif=False,
        with_bias=kwargs.get("with_bias", False),
        bias_ckpt=kwargs.get("bias_ckpt", None),
    )
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def get_finetune_motif_chrombpnet_bias(pretrained=False, **kwargs):
    model = GETFinetuneChrombpNetBias(
        num_motif=128,
        motif_dim=128,
        embed_dim=128,
        motif_prior=False,
        learnable_motif=True,
    )
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

