import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from get_model.model.modules import *
from get_model.model.motif import parse_meme_file
from get_model.model.pooling import (ATACSplitPool, ATACSplitPoolMaxNorm,
                                     ConvPool, SplitPool)
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


class GETPretrainMaxNorm(BaseGETModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.num_regions = cfg.num_regions
        self.num_motif = cfg.num_motif
        self.motif_dim = cfg.motif_dim
        self.num_res_block = cfg.num_res_block
        self.motif_prior = cfg.motif_prior
        self.embed_dim = cfg.embed_dim
        self.num_layers = cfg.num_layers
        self.d_model = cfg.d_model
        self.nhead = cfg.nhead
        self.dropout = cfg.dropout
        self.output_dim = cfg.output_dim
        self.pos_emb_components = cfg.pos_emb_components
        self.flash_attn = cfg.flash_attn

        self.motif_scanner = hydra.utils.instantiate(cfg.motif_scanner)
        self.atac_attention = hydra.utils.instantiate(cfg.atac_attention)
        self.region_embed = hydra.utils.instantiate(cfg.region_embed)
        self.pos_embed = nn.ModuleList([
            hydra.utils.instantiate(cfg.pos_embed) for _ in range(len(self.pos_emb_components))
        ])
        self.encoder = hydra.utils.instantiate(cfg.encoder)

        self.head_mask = nn.Linear(self.d_model, self.output_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    def forward(self, peak_seq, atac, mask, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std):
        x = self.motif_scanner(peak_seq)
        x = x - motif_mean_std[:, 0, :].unsqueeze(1)
        x = x / motif_mean_std[:, 1, :].unsqueeze(1)
        x = F.relu(x)
        x_region = self.split_pool(x, chunk_size, n_peaks, max_n_peaks)
        x_original = self.atac_attention(x, x_region, atac, chunk_size, n_peaks, max_n_peaks)
        x = self.region_embed(x_original)

        B, N, C = x_original.shape
        mask_token = self.mask_token.expand(B, N, -1)
        w = mask.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        for pos_emb_component in self.pos_embed:
            x = pos_emb_component(x)

        x, _ = self.encoder(x, mask=padding_mask)
        x_masked = self.head_mask(x)

        return x_masked, None, x_original


@hydra.main(config_path="config", config_name="model/pretrain")
def get_model(cfg: DictConfig):
    print(f"Creating model: {cfg.model}")
    model = hydra.utils.instantiate(cfg.model)

    if cfg.finetune:
        checkpoint_model = load_checkpoint(cfg.finetune, cfg.model_key)
        state_dict = model.state_dict()
        remove_keys(checkpoint_model, state_dict)
        checkpoint_model = rename_keys(checkpoint_model)
        load_state_dict(model, checkpoint_model, prefix=cfg.model_prefix)

    freeze_layers(model, last_layer=cfg.last_layer, freeze_atac_attention=cfg.freeze_atac_attention)

    return model