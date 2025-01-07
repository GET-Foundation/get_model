import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchmetrics
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig
from torch.nn.init import trunc_normal_

from get_model.model.modules import (ATACHead, ATACHeadConfig, ContactMapHead, ContactMapHeadConfig, ConvBlock, DistanceContactHead, DistanceContactHeadConfig, HiCHead, HiCHeadConfig,
                                     ATACSplitPool,
                                     ATACSplitPoolConfig, ATACSplitPoolMaxNorm,
                                     ATACSplitPoolMaxNormConfig, BaseConfig,
                                     BaseModule, ConvPool, ConvPoolConfig,
                                     ExpressionHead, ExpressionHeadConfig,
                                     MotifScanner, MotifScannerConfig,
                                     RegionEmbed, RegionEmbedConfig, SplitPool, SplitPoolConfig)
from get_model.model.position_encoding import AbsolutePositionalEncoding
from get_model.model.transformer import GETTransformer, GETTransformerWithContactMap, GETTransformerWithContactMapAxial, GETTransformerWithContactMapOE


class MNLLLoss(nn.Module):
    def __init__(self):
        """
From @jmschrei/bpnet-lite
A loss function based on the multinomial negative log-likelihood.

    This loss function takes in a tensor of normalized log probabilities such
    that the sum of each row is equal to 1 (e.g. from a log softmax) and
    an equal sized tensor of true counts and returns the probability of
    observing the true counts given the predicted probabilities under a
    multinomial distribution. Can accept tensors with 2 or more dimensions
    and averages over all except for the last axis, which is the number
    of categories.

    Adapted from Alex Tseng.

    Parameters
    ----------
    logps: torch.tensor, shape=(n, ..., L)
            A tensor with `n` examples and `L` possible categories.

    true_counts: torch.tensor, shape=(n, ..., L)
            A tensor with `n` examples and `L` possible categories.

    Returns
    -------
    loss: float
            The multinomial log likelihood loss of the true counts given the
            predicted probabilities, averaged over all examples and all other
            dimensions.
        """
        super().__init__()

    def forward(self, logps, true_counts):
        log_fact_sum = torch.lgamma(torch.sum(true_counts, dim=-1) + 1)
        log_prod_fact = torch.sum(torch.lgamma(true_counts + 1), dim=-1)
        log_prod_exp = torch.sum(true_counts * logps, dim=-1)
        return (-log_fact_sum + log_prod_fact - log_prod_exp).mean()


class log1pMSELoss(nn.Module):
    def __init__(self):
        """
From @jmschrei/bpnet-lite
A MSE loss on the log(x+1) of the inputs.

    This loss will accept tensors of predicted counts and a vector of true
    counts and return the MSE on the log of the labels. The squared error
    is calculated for each position in the tensor and then averaged, regardless
    of the shape.

    Note: The predicted counts are in log space but the true counts are in the
    original count space.

    Parameters
    ----------
    log_predicted_counts: torch.tensor, shape=(n, ...)
            A tensor of log predicted counts where the first axis is the number of
            examples. Important: these values are already in log space.

    true_counts: torch.tensor, shape=(n, ...)
            A tensor of the true counts where the first axis is the number of
            examples.

    Returns
    -------
    loss: torch.tensor, shape=(n, 1)
            The MSE loss on the log of the two inputs, averaged over all examples
            and all other dimensions.
    """
        super().__init__()

    def forward(self, log_predicted_counts, true_counts):
        log_true = torch.log(true_counts + 1)
        return torch.mean(torch.square(log_true - log_predicted_counts))


@dataclass
class LossConfig:
    components: dict = MISSING
    weights: dict = MISSING


@dataclass
class MetricsConfig:
    components: dict = MISSING


@dataclass
class EncoderConfig:
    num_heads: int = MISSING
    embed_dim: int = MISSING
    num_layers: int = MISSING
    drop_path_rate: float = MISSING
    drop_rate: float = MISSING
    attn_drop_rate: float = MISSING
    use_mean_pooling: bool = False
    flash_attn: bool = MISSING


class GETLoss(nn.Module):
    def __init__(self, cfg: LossConfig):
        """
        Initializes the GETLoss class.

        Args:
            cfg (dict or object): The configuration for the loss function. If `cfg` is a dictionary, it should contain
                the names and configurations of multiple loss functions. If `cfg` is an object, it should be a single
                loss function configuration.

        """
        super(GETLoss, self).__init__()
        self.cfg = cfg
        if isinstance(cfg, DictConfig):
            self.losses = {name: (
                component, cfg.weights[f'{name}']) for name, component in cfg.components.items()}
        else:
            self.losses = instantiate(cfg)

    def forward(self, pred, obs):
        """Compute the loss"""
        if isinstance(self.losses, dict):
            return {f"{name}_loss": loss_fn(pred[name], obs[name]) * weight for name, (loss_fn, weight) in self.losses.items()}
        elif isinstance(self.losses, nn.Module):
            return self.losses(pred, obs)

    def freeze_component(self, component_name):
        """Freeze a component of the loss function by set the weight to 0"""
        if component_name in self.losses:
            self.losses[component_name] = (self.losses[component_name][0], 0)
        else:
            raise ValueError(
                f"Component '{component_name}' not found in the loss function.")


class RegressionMetrics(nn.Module):
    def __init__(self, _cfg_: MetricsConfig):
        super(RegressionMetrics, self).__init__()
        self.cfg = _cfg_
        self.metrics = nn.ModuleDict({
            target: nn.ModuleDict({
                metric_name: self._get_metric(metric_name) for metric_name in metric_names
            }) for target, metric_names in _cfg_.components.items()
        })

    def _get_metric(self, metric_name):
        if metric_name == 'pearson':
            return torchmetrics.PearsonCorrCoef()
        elif metric_name == 'spearman':
            return torchmetrics.SpearmanCorrCoef()
        elif metric_name == 'mse':
            return torchmetrics.MeanSquaredError()
        elif metric_name == 'r2':
            return torchmetrics.R2Score()
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")

    def forward(self, _pred_, _obs_):
        """Compute the metrics"""
        batch_size = _pred_[list(_pred_.keys())[0]].shape[0]
        result = {
            target: {
                metric_name: metric(
                    _pred_[target].reshape(-1, 1),
                    _obs_[target].reshape(-1, 1))
                for metric_name, metric in target_metrics.items()
            }
            for target, target_metrics in self.metrics.items()
        }
        # flatten the result
        result = {f"{target}_{metric_name}": result[target][metric_name]
                  for target in result for metric_name in result[target]}
        return result


@dataclass
class BaseGETModelConfig:
    freezed: bool | str = False
    loss: LossConfig = MISSING
    metrics: MetricsConfig = MISSING


class BaseGETModel(BaseModule):
    def __init__(self, cfg: BaseConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.loss = GETLoss(cfg.loss)
        self.metrics = RegressionMetrics(cfg.metrics)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_input(self, batch):
        """Prepare the input for the model"""
        raise NotImplementedError

    def forward(self, batch):
        raise NotImplementedError

    def before_loss(self, output, batch):
        """Prepare the output and target for the loss function
        The goal is to construct either:
        1. pred and obs tensors, in which case a defined loss function is applied to pred and obs
        2. pred: {name: tensor} and obs: {name: tensor}, in which case we will use the
        loss_cfg: {name: loss_fn} to determine the loss function for each name"""
        raise NotImplementedError

    def after_loss(self, loss):
        """Prepare the loss for the optimizer"""
        if isinstance(loss, dict):
            # combine losses
            return sum(loss.values())
        else:
            return loss

    def freeze_layers(self, patterns_to_freeze=None, invert_match=False):
        """
        Freeze layers in a model based on matching patterns.

        Parameters:
        - self (torch.nn.Module): The model whose layers will be frozen.
        - patterns_to_freeze (list of str, optional): A list of string patterns. Layers matching any of these patterns will be frozen.
        - invert_match (bool): If True, layers matching the patterns will remain trainable and all others will be frozen. Default is False.

        If `patterns_to_freeze` is None or an empty list, and `invert_match` is False, no layers will be frozen.
        If `patterns_to_freeze` is None or an empty list, and `invert_match` is True, all layers will be frozen.
        """

        # Ensure there's a list of patterns to check against.
        if patterns_to_freeze is None:
            patterns_to_freeze = []

        for name, param in self.named_parameters():
            # Determine if the current parameter name matches any pattern.
            matches_pattern = any(
                pattern in name for pattern in patterns_to_freeze)

            # Decide whether to freeze based on `invert_match` and if the name matches any pattern.
            should_freeze = matches_pattern if not invert_match else not matches_pattern

            if should_freeze:
                param.requires_grad = False
                logging.debug(f"Freezed weights of {name}")

    def generate_dummy_data(self):
        """Return a dummy input for the model"""
        raise NotImplementedError

    def get_layer(self, layer_name):
        if hasattr(self, layer_name):
            return getattr(self, layer_name)
        else:
            raise ValueError(f"Layer '{layer_name}' not found in the model.")

    def get_layer_names(self):
        return list(self._modules.keys())

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}


@dataclass
class GETPretrainModelConfig(BaseGETModelConfig):
    num_regions: int = 10
    num_motif: int = 637
    embed_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.1
    output_dim: int = 800
    flash_attn: bool = False
    pool_method: str = 'mean'
    motif_scanner: MotifScannerConfig = field(
        default_factory=MotifScannerConfig)
    atac_attention: ATACSplitPoolConfig = field(
        default_factory=ATACSplitPoolConfig)
    region_embed: RegionEmbedConfig = field(default_factory=RegionEmbedConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    head_mask: dict = field(default_factory=lambda: {
                            'in_features': 768, 'out_features': 800})
    mask_token: dict = field(default_factory=lambda: {
                             'embed_dim': 768, 'std': 0.02})


class GETPretrain(BaseGETModel):
    def __init__(self, cfg: GETPretrainModelConfig):
        super().__init__(cfg)
        self.motif_scanner = MotifScanner(cfg.motif_scanner)
        self.atac_attention = ATACSplitPool(cfg.atac_attention)
        self.region_embed = RegionEmbed(cfg.region_embed)
        self.encoder = GETTransformer(**cfg.encoder)
        self.head_mask = nn.Linear(**cfg.head_mask)
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, cfg.mask_token.embed_dim))
        trunc_normal_(self.mask_token, std=cfg.mask_token.std)

        self.apply(self._init_weights)

    def get_input(self, batch):
        return {'sample_peak_sequence': batch['sample_peak_sequence'],
                'sample_track': batch['sample_track'],
                'loss_mask': batch['loss_mask'],
                'padding_mask': batch['padding_mask'],
                'chunk_size': batch['chunk_size'],
                'n_peaks': batch['n_peaks'],
                'max_n_peaks': batch['max_n_peaks'],
                'motif_mean_std': batch['motif_mean_std']}

    def forward(self, sample_peak_sequence, sample_track, loss_mask, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std):
        x = self.motif_scanner(sample_peak_sequence, motif_mean_std)
        # x_region = self.split_pool(x, chunk_size, n_peaks, max_n_peaks)
        x_original = self.atac_attention(
            x, sample_track, chunk_size, n_peaks, max_n_peaks)
        x = self.region_embed(x_original)
        B, N, C = x_original.shape
        mask_token = self.mask_token.expand(B, N, -1)
        w = loss_mask.type_as(mask_token)
        x = x * (1 - w) + mask_token * w
        x, _ = self.encoder(x, mask=padding_mask)
        x_masked = self.head_mask(x)
        return x_masked, x_original, loss_mask

    def before_loss(self, output, batch):
        x_masked, x_original, loss_mask = output
        pred = {'masked': x_masked * loss_mask}
        obs = {'masked': x_original * loss_mask}
        return pred, obs

    def generate_dummy_data(self):
        B, R, L = 2, 10, 200
        return {
            'sample_peak_sequence': torch.randint(0, 4, (B, R * L, 4)).float(),
            'sample_track': torch.randn(B, R*L).float().abs(),
            'loss_mask': torch.randint(0, 2, (B, R)).bool().unsqueeze(-1),
            'padding_mask': torch.randint(0, 2, (B, R)).bool(),
            'chunk_size':  torch.Tensor(([L]*R + [0]) * B).int().tolist(),
            'n_peaks': (torch.zeros(B,) + R).int(),
            'max_n_peaks': R,
            'motif_mean_std': torch.randn(B, 2, 639).abs().float()
        }


@dataclass
class GETPretrainMaxNormModelConfig(GETPretrainModelConfig):
    atac_attention: ATACSplitPoolMaxNormConfig = field(
        default_factory=ATACSplitPoolMaxNormConfig)


class GETPretrainMaxNorm(GETPretrain):
    def __init__(self, cfg: GETPretrainMaxNormModelConfig):
        super().__init__(cfg)
        self.atac_attention = ATACSplitPoolMaxNorm(cfg.atac_attention)


@dataclass
class GETFinetuneModelConfig(BaseGETModelConfig):
    motif_scanner: MotifScannerConfig = field(
        default_factory=MotifScannerConfig)
    atac_attention: ATACSplitPoolConfig = field(
        default_factory=ATACSplitPoolConfig)
    region_embed: RegionEmbedConfig = field(default_factory=RegionEmbedConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    head_exp: ExpressionHeadConfig = field(
        default_factory=ExpressionHeadConfig)
    use_atac: bool = False
    final_bn: bool = False


class GETFinetune(BaseGETModel):
    def __init__(self, cfg: GETFinetuneModelConfig):
        super().__init__(cfg)
        self.motif_scanner = MotifScanner(cfg.motif_scanner)
        self.atac_attention = ATACSplitPool(cfg.atac_attention)
        self.region_embed = RegionEmbed(cfg.region_embed)
        self.encoder = GETTransformer(**cfg.encoder)
        self.head_exp = ExpressionHead(cfg.head_exp)

        self.apply(self._init_weights)

    def get_input(self, batch):
        return {
            'sample_peak_sequence': batch['sample_peak_sequence'],
            'sample_track': batch['sample_track'],
            'padding_mask': batch['padding_mask'],
            'chunk_size': batch['chunk_size'],
            'n_peaks': batch['n_peaks'],
            'max_n_peaks': batch['max_n_peaks'],
            'motif_mean_std': batch['motif_mean_std'],
        }

    def forward(self, sample_peak_sequence, sample_track, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std):
        x = self.motif_scanner(sample_peak_sequence, motif_mean_std)
        x_original = self.atac_attention(
            x, sample_track, chunk_size, n_peaks, max_n_peaks)
        x = self.region_embed(x_original)

        x, _ = self.encoder(x, mask=padding_mask)
        exp = F.softplus(self.head_exp(x))
        return exp

    def before_loss(self, output, batch):
        pred = {'exp': output}
        obs = {'exp': batch['exp_label']}
        return pred, obs

    def generate_dummy_data(self):
        B, R, L = 2, 10, 200
        return {
            'sample_peak_sequence': torch.randint(0, 4, (B, R * L, 4)).float(),
            'sample_track': torch.randn(B, R*L).float().abs(),
            'padding_mask': torch.randint(0, 2, (B, R)).bool(),
            'chunk_size':  torch.Tensor(([L]*R + [0]) * B).int().tolist(),
            'n_peaks': (torch.zeros(B,) + R).int(),
            'max_n_peaks': R,
            'motif_mean_std': torch.randn(B, 2, 639).abs().float(),
        }


@dataclass
class GETFinetuneGBMConfig(GETFinetuneModelConfig):
    head_atac: ExpressionHeadConfig = field(
        default_factory=ExpressionHeadConfig)


class GETFinetuneGBM(GETFinetune):
    def __init__(self, cfg: GETFinetuneGBMConfig):
        super().__init__(cfg)
        self.head_atac = ATACHead(cfg.head_atac)

    def forward(self, sample_peak_sequence, sample_track, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std):
        x = self.motif_scanner(sample_peak_sequence, motif_mean_std)
        x_original = self.atac_attention(
            x, sample_track, chunk_size, n_peaks, max_n_peaks)
        x = self.region_embed(x_original)

        x, _ = self.encoder(x, mask=padding_mask)
        exp = F.softplus(self.head_exp(x))
        atac = F.softplus(self.head_atac(x_original))
        return exp, atac

    def before_loss(self, output, batch):
        pred = {'exp': output[0], 'atac': output[1]}
        obs = {'exp': batch['exp_label'], 'atac': batch['atpm'].unsqueeze(-1)}
        return pred, obs


@dataclass
class GETFinetuneMaxNormModelConfig(GETFinetuneModelConfig):
    atac_attention: ATACSplitPoolMaxNormConfig = MISSING


class GETFinetuneMaxNorm(GETFinetune):
    def __init__(self, cfg: GETFinetuneMaxNormModelConfig):
        super().__init__(cfg)
        self.atac_attention = ATACSplitPoolMaxNorm(cfg.atac_attention)


@dataclass
class GETRegionPretrainModelConfig(BaseGETModelConfig):
    region_embed: RegionEmbedConfig = field(default_factory=RegionEmbedConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    head_mask: dict = field(default_factory=lambda: {
                            'in_features': 768, 'out_features': 283})
    mask_token: dict = field(default_factory=lambda: {
                             'embed_dim': 768, 'std': 0.02})


class GETRegionPretrain(BaseGETModel):
    def __init__(self, cfg: GETRegionPretrainModelConfig):
        super().__init__(cfg)
        self.region_embed = RegionEmbed(cfg.region_embed)
        self.encoder = GETTransformer(**cfg.encoder)
        self.head_mask = nn.Linear(**cfg.head_mask)
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, cfg.mask_token.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        trunc_normal_(self.mask_token, std=cfg.mask_token.std)

        self.apply(self._init_weights)

    def get_input(self, batch):
        return {
            'region_motif': batch['region_motif'],
            'mask': batch['mask'].unsqueeze(-1).bool()
        }

    def forward(self, region_motif, mask):
        x = self.region_embed(region_motif)
        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        mask_token = self.mask_token.expand(B, N, -1)
        w = mask.type_as(mask_token)
        x = x * (1 - w) + mask_token * w
        x = torch.cat((cls_tokens, x), dim=1)
        x, _ = self.encoder(x)
        x = x[:, 1:][mask.squeeze()].reshape( -1, C)
        x_masked = self.head_mask(x)
        return x_masked, region_motif, mask

    def before_loss(self, output, batch):
        x_masked, x_original, loss_mask = output
        B, _, C = x_original.shape
        pred = {'masked': x_masked}
        obs = {'masked': x_original[loss_mask.squeeze()].reshape(-1, C)}
        return pred, obs

    def generate_dummy_data(self):
        B, R, M = 2, 900, 283
        return {
            'region_motif': torch.randn(B, R, M).float().abs(),
        }


@dataclass
class GETRegionFinetuneModelConfig(BaseGETModelConfig):
    region_embed: RegionEmbedConfig = field(default_factory=RegionEmbedConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    head_exp: ExpressionHeadConfig = field(
        default_factory=ExpressionHeadConfig)
    use_atac: bool = False


class GETRegionFinetune(BaseGETModel):
    def __init__(self, cfg: GETRegionFinetuneModelConfig):
        super().__init__(cfg)
        self.region_embed = RegionEmbed(cfg.region_embed)
        self.encoder = GETTransformer(**cfg.encoder)
        self.head_exp = ExpressionHead(cfg.head_exp)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.apply(self._init_weights)

    def get_input(self, batch, perturb=False):
        return {
            'region_motif': batch['region_motif'],
        }

    def forward(self, region_motif):

        x = self.region_embed(region_motif)
        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x, _ = self.encoder(x)
        x = x[:, 1:]
        exp = nn.Softplus()(self.head_exp(x))
        return exp

    def before_loss(self, output, batch):

        pred = {'exp': output}
        obs = {'exp': batch['exp_label']}
        return pred, obs

    def generate_dummy_data(self):
        B, R, M = 2, 900, 283
        return {
            'region_motif': torch.randn(B, R, M).float().abs(),
        }


class GETRegionFinetunePositional(GETRegionFinetune):
    def __init__(self, cfg: GETRegionFinetuneModelConfig):
        super().__init__(cfg)
        self.region_embed = RegionEmbed(cfg.region_embed)
        self.pos_embed = AbsolutePositionalEncoding(
            cfg.region_embed.embed_dim, dropout=0.1, max_len=1000)
        self.encoder = GETTransformer(**cfg.encoder)
        self.head_exp = ExpressionHead(cfg.head_exp)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.apply(self._init_weights)

    def get_input(self, batch, perturb=False):
        return {
            'region_motif': batch['region_motif'],
        }

    def forward(self, region_motif):

        x = self.region_embed(region_motif)
        x = self.pos_embed(x)
        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x, _ = self.encoder(x)
        x = x[:, 1:]
        exp = nn.Softplus()(self.head_exp(x))
        return exp

    def before_loss(self, output, batch):

        pred = {'exp': output}
        obs = {'exp': batch['exp_label']}
        return pred, obs


@dataclass
class MLPRegionFinetuneModelConfig(BaseGETModelConfig):
    use_atac: bool = False
    input_dim: int = 283
    output_dim: int = 2


class MLPRegionFinetune(GETRegionFinetune):
    def __init__(self, cfg: MLPRegionFinetuneModelConfig):
        super(GETRegionFinetune, self).__init__(cfg)
        self.linear1 = torch.nn.Linear(cfg.input_dim, 512)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(512, 256)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(256, cfg.output_dim)

    def forward(self, region_motif):
        x = self.linear1(region_motif)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = torch.nn.Softplus()(x)
        return x


@dataclass
class GETRegionFinetuneATACModelConfig(BaseGETModelConfig):
    region_embed: RegionEmbedConfig = field(default_factory=RegionEmbedConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    head_exp: ExpressionHeadConfig = field(
        default_factory=ExpressionHeadConfig)


class GETRegionFinetuneATAC(BaseGETModel):
    def __init__(self, cfg: GETRegionFinetuneModelConfig):
        super().__init__(cfg)
        self.region_embed = RegionEmbed(cfg.region_embed)
        self.encoder = GETTransformer(**cfg.encoder)
        self.head_exp = ExpressionHead(cfg.head_exp)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.apply(self._init_weights)

    def get_input(self, batch):
        input = batch['region_motif'].clone()
        input[:, :, -1] = 1
        return {
            'region_motif': input,
        }

    def forward(self, region_motif):

        x = self.region_embed(region_motif)
        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x, _ = self.encoder(x)
        x = x[:, 1:]
        atpm = nn.Softplus()(self.head_exp(x))
        return atpm

    def before_loss(self, output, batch):

        pred = {'atpm': output}
        obs = {'atpm': batch['region_motif'][:, :, -1].unsqueeze(-1)}
        return pred, obs

    def generate_dummy_data(self):
        B, R, M = 2, 900, 283
        return {
            'region_motif': torch.randn(B, R, M).float().abs(),
        }


@dataclass
class GETRegionFinetuneHiCModelConfig(BaseGETModelConfig):
    region_embed: RegionEmbedConfig = field(default_factory=RegionEmbedConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    head_hic: HiCHeadConfig = field(
        default_factory=HiCHeadConfig)


class GETRegionFinetuneHiC(BaseGETModel):
    def __init__(self, cfg: GETRegionFinetuneHiCModelConfig):
        super().__init__(cfg)
        self.region_embed = RegionEmbed(cfg.region_embed)
        self.encoder = GETTransformer(**cfg.encoder)
        self.head_hic = HiCHead(cfg.head_hic)

        self.apply(self._init_weights)

    def get_input(self, batch):
        return {
            'region_motif': batch['region_motif'],
        }

    def forward(self, region_motif):

        x = self.region_embed(region_motif)

        x, _ = self.encoder(x)
        hic = self.head_hic(x)
        return hic

    def before_loss(self, output, batch):

        pred = {'hic': output.squeeze(-1)}
        obs = {'hic': batch['hic_matrix'].float()}
        return pred, obs

    def generate_dummy_data(self):
        B, R, M = 2, 900, 283
        return {
            'region_motif': torch.randn(B, R, M).float().abs(),
        }


@dataclass
class DistanceContactMapModelConfig(BaseGETModelConfig):
    pass


class DistanceContactMap(BaseGETModel):
    """A simple and small Conv2d model to predict the contact map from the log distance map.
    The output has the same shape as the input distance map.
    """

    def __init__(self, cfg: DistanceContactMapModelConfig):
        super().__init__(cfg)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, 3, padding=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, distance_map):
        x = F.gelu(self.conv1(distance_map))
        x = F.gelu(self.conv2(x))
        x = self.conv3(x)
        hic = F.softplus(x)
        return hic

    def get_input(self, batch):
        peak_coord = batch['peak_coord']
        peak_coord_mean = peak_coord[:, :, 0]
        # pair-wise distance using torch
        # Add new dimensions to create column and row vectors
        peak_coord_mean_col = peak_coord_mean.unsqueeze(
            2)  # Adds a new axis (column vector)
        peak_coord_mean_row = peak_coord_mean.unsqueeze(
            1)  # Adds a new axis (row vector)

        # Compute the pairwise difference
        distance = torch.log10(
            (peak_coord_mean_col - peak_coord_mean_row).abs() + 1).unsqueeze(1)
        return {
            'distance_map': distance,
        }

    def before_loss(self, output, batch):
        pred = {'hic': output.squeeze(1)}
        obs = {'hic': batch['hic_matrix'].float()}
        # if a element in batch['hic_matrix'].sum(1).sum(1) == 0, we should ignore the loss for that element
        if (batch['hic_matrix'].sum(1).sum(1) == 0).any():
            mask = batch['hic_matrix'].sum(1).sum(1) != 0
            obs['hic'][mask] = pred['hic'][mask].detach()

        return pred, obs

    def generate_dummy_data(self):
        B, R = 2, 900
        return {
            'distance_map': torch.randn(B, 1, R, R).float(),
        }


@dataclass
class GETRegionFinetuneExpHiCABCConfig(BaseGETModelConfig):
    region_embed: RegionEmbedConfig = field(default_factory=RegionEmbedConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    head_exp: ExpressionHeadConfig = field(
        default_factory=ExpressionHeadConfig)
    header_hic: ContactMapHeadConfig = field(
        default_factory=ContactMapHeadConfig)
    header_abc: ContactMapHeadConfig = field(
        default_factory=ContactMapHeadConfig)
    distance_contact_map: DistanceContactHeadConfig = field(
        default_factory=DistanceContactHeadConfig)


class GETRegionFinetuneExpHiCABC(BaseGETModel):
    def __init__(self, cfg: GETRegionFinetuneExpHiCABCConfig):
        super().__init__(cfg)
        self.region_embed = RegionEmbed(cfg.region_embed)
        self.encoder = GETTransformerWithContactMap(**cfg.encoder)
        self.head_exp = ExpressionHead(cfg.head_exp)
        self.hic_header = ContactMapHead(cfg.header_hic)
        self.abc_header = ContactMapHead(cfg.header_abc)
        self.distance_contact_map = DistanceContactHead(
            cfg.distance_contact_map)
        self.distance_contact_map.eval()
        self.proj_distance = nn.Linear(cfg.embed_dim, 128)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        logging.info(
            f"GETRegionFinetuneExpHiCABC can only be used with quantitative_atac=True in order to generate the ABC score.")
        self.apply(self._init_weights)

    def get_input(self, batch, perturb=False):
        peak_coord = batch['peak_coord']
        peak_coord_mean = peak_coord[:, :, 0]
        # pair-wise distance using torch
        # Add new dimensions to create column and row vectors
        peak_coord_mean_col = peak_coord_mean.unsqueeze(
            2)  # Adds a new axis (column vector)
        peak_coord_mean_row = peak_coord_mean.unsqueeze(
            1)  # Adds a new axis (row vector)

        # Compute the pairwise difference
        distance = torch.log10(
            (peak_coord_mean_col - peak_coord_mean_row).abs() + 1).unsqueeze(1)
        region_model = batch['region_motif'].clone()
        batch['distance_map'] = distance
        # region_model[:, :, -1] = 1 # set atac to binary
        return {
            'region_motif': region_model,
            'distance_map': distance,
        }

    def forward(self, region_motif, distance_map):
        x = self.region_embed(region_motif)
        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x, fused_distance_map, _ = self.encoder(x, distance_map)
        x = x[:, 1:]
        exp = nn.Softplus()(self.head_exp(x))
        fused_distance_map = self.proj_distance(
            fused_distance_map).transpose(1, 3).transpose(2, 3)
        hic = self.hic_header(fused_distance_map).squeeze(1)
        abc = self.abc_header(fused_distance_map).squeeze(1)
        return exp, hic, abc

    def before_loss(self, output, batch):
        exp, hic_contact_map, abc_contact_map = output
        atac = batch['region_motif'][:, :, -1]
        # outer sum of atac and itself
        atac = torch.sqrt(atac.unsqueeze(1) * atac.unsqueeze(2))
        # test if batch['hic_matrix'] has shape and not a scalar
        if len(batch['hic_matrix'][0].shape) >= 2:
            hic = batch['hic_matrix'].float()
            real_hic = True
        else:
            logging.info(
                f"batch['hic_matrix'] is not a matrix, using the distance contact map instead.")
            hic = self.distance_contact_map(
                batch['distance_map']).detach().squeeze(1)
            real_hic = False
        # abc = atac * hic
        pred = {
            'exp': exp,
            'hic': hic_contact_map,
            # 'abc': abc_contact_map,
        }
        obs = {
            'exp': batch['exp_label'],
            'hic': hic,
            # 'abc': abc,
        }
        if real_hic:
            mask = hic.sum(1).sum(1) == 0
            obs['hic'][mask] = pred['hic'][mask].detach()
            # obs['abc'][mask] = pred['abc'][mask].detach()
        return pred, obs

    def generate_dummy_data(self):
        B, R, M = 2, 900, 283
        return {
            'region_motif': torch.randn(B, R, M).float().abs(),
            'distance_map': torch.randn(B, R, R).float(),
        }


@dataclass
class GETRegionFinetuneHiCOEConfig(BaseGETModelConfig):
    region_embed: RegionEmbedConfig = field(default_factory=RegionEmbedConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    header_hic: ContactMapHeadConfig = field(
        default_factory=ContactMapHeadConfig)
    distance_contact_map: DistanceContactHeadConfig = field(
        default_factory=DistanceContactHeadConfig)


class GETRegionFinetuneHiCOE(BaseGETModel):
    def __init__(self, cfg: GETRegionFinetuneHiCOEConfig):
        super().__init__(cfg)
        self.region_embed = RegionEmbed(cfg.region_embed)
        self.encoder = GETTransformerWithContactMapOE(**cfg.encoder)
        self.hic_header = ContactMapHead(cfg.header_hic, activation='none')
        self.distance_contact_map = DistanceContactHead(
            cfg.distance_contact_map)
        self.distance_contact_map.eval()
        self.proj_distance = nn.Linear(cfg.embed_dim+1, 128)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        logging.info(
            f"GETRegionFinetuneExpHiCABC can only be used with quantitative_atac=True in order to generate the ABC score.")
        self.apply(self._init_weights)

    def get_input(self, batch, perturb=False):
        peak_coord = batch['peak_coord']
        peak_length = peak_coord[:, :, 1] - peak_coord[:, :, 0]
        peak_coord_mean = peak_coord[:, :, 0]
        # pair-wise distance using torch
        # Add new dimensions to create column and row vectors
        peak_coord_mean_col = peak_coord_mean.unsqueeze(
            2)  # Adds a new axis (column vector)
        peak_coord_mean_row = peak_coord_mean.unsqueeze(
            1)  # Adds a new axis (row vector)

        # Compute the pairwise difference
        distance = torch.log10(
            (peak_coord_mean_col - peak_coord_mean_row).abs() + 1).unsqueeze(1)
        region_model = batch['region_motif'].clone()
        batch['distance_map'] = distance
        # region_model[:, :, -1] = 1 # set atac to binary
        return {
            'region_motif': region_model,
            'distance_map': distance,
            'peak_length': peak_length/1000
        }

    def forward(self, region_motif, distance_map, peak_length):
        # concat peak_length to the region_motif
        region_motif  = torch.cat([region_motif, peak_length.unsqueeze(-1)], dim=-1)
        x = self.region_embed(region_motif)
        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x, fused_distance_map, _ = self.encoder(x, distance_map)
        # x = x[:, 1:]
        fused_distance_map = self.proj_distance(
            fused_distance_map).transpose(1, 3).transpose(2, 3)

        hic = self.hic_header(fused_distance_map).squeeze(1)
        return hic

    def before_loss(self, output, batch):
        hic_contact_map = output
        if len(batch['hic_matrix'][0].shape) >= 2:
            hic = batch['hic_matrix'].to(torch.float16)
            real_hic = True
        else:
            logging.info(
                f"batch['hic_matrix'] is not a matrix, using the distance contact map instead.")
            hic = self.distance_contact_map(
                batch['distance_map']).detach().squeeze(1)
            real_hic = False
        pred = {
            'hic': hic_contact_map,
        }
        obs = {
            'hic': hic.float(),
        }
        if real_hic:
            mask = (hic==0)
            obs['hic'][mask] = pred['hic'][mask].detach()
        return pred, obs

    def generate_dummy_data(self):
        B, R, M = 2, 900, 283
        return {
            'region_motif': torch.randn(B, R, M).float().abs(),
            'distance_map': torch.randn(B, R, R).float(),
        }




@dataclass
class GETRegionFinetuneExpHiCAxialModelConfig(BaseGETModelConfig):
    region_embed: RegionEmbedConfig = field(default_factory=RegionEmbedConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    head_exp: ExpressionHeadConfig = field(
        default_factory=ExpressionHeadConfig)
    header_hic: ContactMapHeadConfig = field(
        default_factory=ContactMapHeadConfig)
    header_abc: ContactMapHeadConfig = field(
        default_factory=ContactMapHeadConfig)
    distance_contact_map: DistanceContactHeadConfig = field(
        default_factory=DistanceContactHeadConfig)


class GETRegionFinetuneExpHiCAxial(BaseGETModel):
    def __init__(self, cfg: GETRegionFinetuneExpHiCAxialModelConfig):
        super().__init__(cfg)
        self.region_embed = RegionEmbed(cfg.region_embed)
        self.encoder = GETTransformerWithContactMapAxial(**cfg.encoder)
        self.head_exp = ExpressionHead(cfg.head_exp)
        # self.hic_header = ContactMapHead(cfg.header_hic)
        # self.abc_header = ContactMapHead(cfg.header_abc)
        self.distance_contact_map = DistanceContactHead(
            cfg.distance_contact_map)
        self.distance_contact_map.eval()
        self.proj_distance = nn.Linear(cfg.embed_dim, 1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        logging.info(
            f"GETRegionFinetuneExpHiCABC can only be used with quantitative_atac=True in order to generate the ABC score.")
        self.apply(self._init_weights)

    def get_input(self, batch, perturb=False):
        peak_coord = batch['peak_coord']
        peak_coord_mean = peak_coord[:, :, 0]
        # pair-wise distance using torch
        # Add new dimensions to create column and row vectors
        peak_coord_mean_col = peak_coord_mean.unsqueeze(
            2)  # Adds a new axis (column vector)
        peak_coord_mean_row = peak_coord_mean.unsqueeze(
            1)  # Adds a new axis (row vector)

        # Compute the pairwise difference
        distance = torch.log10(
            (peak_coord_mean_col - peak_coord_mean_row).abs() + 1).unsqueeze(1)
        region_model = batch['region_motif'].clone()
        batch['distance_map'] = distance
        # region_model[:, :, -1] = 1 # set atac to binary
        return {
            'region_motif': region_model,
            'distance_map': distance,
        }

    def forward(self, region_motif, distance_map):
        x = self.region_embed(region_motif)
        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x, fused_distance_map, _ = self.encoder(x, distance_map)
        x = x[:, 1:]
        exp = nn.Softplus()(self.head_exp(x))
        fused_distance_map = self.proj_distance(
            fused_distance_map)  # .transpose(1, 3).transpose(2, 3)
        # hic = self.hic_header(fused_distance_map).squeeze(1)
        # abc = self.abc_header(fused_distance_map).squeeze(1)
        return exp, fused_distance_map.squeeze(3)

    def before_loss(self, output, batch):
        exp, hic_contact_map = output
        atac = batch['region_motif'][:, :, -1]
        # outer sum of atac and itself
        atac = torch.sqrt(atac.unsqueeze(1) * atac.unsqueeze(2))
        # test if batch['hic_matrix'] has shape and not a scalar
        if len(batch['hic_matrix'][0].shape) >= 2:
            hic = batch['hic_matrix'].float()
            real_hic = True
        else:
            logging.info(
                f"batch['hic_matrix'] is not a matrix, using the distance contact map instead.")
            hic = self.distance_contact_map(
                batch['distance_map']).detach().squeeze(1)
            real_hic = False
        # abc = atac * hic
        pred = {
            'exp': exp,
            'hic': hic_contact_map,
            # 'abc': abc_contact_map,
        }
        obs = {
            'exp': batch['exp_label'],
            'hic': hic,
            # 'abc': abc,
        }
        if real_hic:
            mask = hic.sum(1).sum(1) == 0
            # obs['abc'][mask] = pred['abc'][mask].detach()
            # obs['abc'][0, :] = pred['abc'][0, :].detach()
            # obs['abc'][-1, :] = pred['abc'][-1, :].detach()
            # obs['abc'][:, 0] = pred['abc'][:, 0].detach()
            # obs['abc'][:, -1] = pred['abc'][:, -1].detach()
            obs['hic'][mask] = pred['hic'][mask].detach().float()
            # obs['hic'][0, :] = pred['hic'][0, :].detach()
            # obs['hic'][-1, :] = pred['hic'][-1, :].detach()
            # obs['hic'][:, 0] = pred['hic'][:, 0].detach()
            # obs['hic'][:, -1] = pred['hic'][:, -1].detach()
        return pred, obs

    def generate_dummy_data(self):
        B, R, M = 2, 900, 283
        return {
            'region_motif': torch.randn(B, R, M).float().abs(),
            'distance_map': torch.randn(B, R, R).float(),
        }


@dataclass
class GETChrombpNetBiasModelConfig(BaseGETModelConfig):
    motif_scanner: MotifScannerConfig = MISSING
    atac_attention: ConvPoolConfig = MISSING


class GETChrombpNetBias(BaseGETModel):
    def __init__(self, cfg: GETChrombpNetBiasModelConfig):
        super().__init__(cfg)
        self.motif_scanner = MotifScanner(cfg.motif_scanner)
        self.atac_attention = ConvPool(cfg.atac_attention)

        self.apply(self._init_weights)

    def get_input(self, batch):
        return {
            'sample_peak_sequence': batch['sample_peak_sequence'],
            'chunk_size': batch['chunk_size'],
            'n_peaks': batch['n_peaks'],
            'max_n_peaks': batch['max_n_peaks'],
            'motif_mean_std': batch['motif_mean_std'],
        }

    def crop_output(self, aprofile, aprofile_target, B, R, target_length=1000):
        # crop aprofile to center 1000bp, assume the input is (B, R, L)
        aprofile = aprofile.reshape(B, R, -1)
        aprofile_target = aprofile_target.reshape(B, R, -1)
        if aprofile.shape[2] != aprofile_target.shape[2]:
            current_length = aprofile.shape[2]
            diff_length = current_length - target_length
            assert diff_length % 2 == 0
            crop_size = diff_length // 2
            aprofile = aprofile[:, :, crop_size:crop_size+target_length]
            aprofile_target = aprofile_target.reshape(B, R, -1)
            diff_length = aprofile_target.shape[2] - target_length
            assert diff_length % 2 == 0
            crop_size = diff_length // 2
            aprofile_target = aprofile_target[:,
                                              :, crop_size:crop_size+target_length]
        return aprofile, aprofile_target

    def forward(self, sample_peak_sequence, chunk_size, n_peaks, max_n_peaks, motif_mean_std):
        x = self.motif_scanner(sample_peak_sequence)
        atpm, aprofile = self.atac_attention(
            x, chunk_size, n_peaks, max_n_peaks)
        return {'atpm': atpm, 'aprofile': aprofile}

    def before_loss(self, output, batch):
        pred = output
        B, R = pred['atpm'].shape
        obs = {'aprofile': torch.log2(batch['sample_track']+1)}
        pred['aprofile'] = F.softplus(pred['aprofile'])
        pred['aprofile'], obs['aprofile'] = self.crop_output(
            pred['aprofile'], obs['aprofile'], B, R)
        obs_atpm = batch['sample_track'].sum(dim=1).unsqueeze(-1)
        obs_atpm = torch.log10(obs_atpm*1e5/batch['metadata'][0]['libsize']+1).detach()
        obs['atpm'] = obs_atpm
        pred_atpm = pred['aprofile'].sum(dim=2).unsqueeze(-1)
        pred_atpm = torch.log10(pred_atpm*1e5/batch['metadata'][0]['libsize']+1)
        pred = {'atpm': pred_atpm,
                'aprofile': torch.log2(pred['aprofile']+1)}
        return pred, obs

    def generate_dummy_data(self):
        B, R, L = 2, 1, 2000
        return {
            'sample_peak_sequence': torch.randint(0, 4, (B, R * L, 4)).float(),
            'chunk_size':  torch.Tensor(([L]*R + [0]) * B).int().tolist(),
            'n_peaks': (torch.zeros(B,) + R).int(),
            'max_n_peaks': R,
            'motif_mean_std': torch.randn(B, 2, 639).abs().float(),
        }



@dataclass
class GETChrombpNetModelConfig(GETChrombpNetBiasModelConfig):
    motif_scanner: MotifScannerConfig = MISSING
    atac_attention: ConvPoolConfig = MISSING
    with_bias: bool = False
    bias_model: GETChrombpNetBiasModelConfig = MISSING
    bias_ckpt: str = None


class GETChrombpNet(GETChrombpNetBias):
    def __init__(self, cfg: GETChrombpNetModelConfig):
        super().__init__(cfg)
        self.with_bias = cfg.with_bias
        if self.with_bias:
            self.bias_model = cfg.bias_model
            if cfg.bias_ckpt is not None:
                checkpoint = torch.load(cfg.bias_ckpt, map_location="cpu")
                if 'model' in checkpoint:
                    checkpoint = checkpoint['model']
                if 'state_dict' in checkpoint:
                    checkpoint = checkpoint['state_dict']
                if 'model.' in list(checkpoint.keys())[0]:
                    checkpoint = {
                        k.replace('model.', ''): v for k, v in checkpoint.items()}
                self.bias_model.load_state_dict(checkpoint)
                for param in self.bias_model.parameters():
                    param.requires_grad = False

        self.apply(self._init_weights)

    def get_input(self, batch, perturb=False):
        result = {
            'sample_peak_sequence': batch['sample_peak_sequence'],
            'chunk_size': batch['chunk_size'],
            'n_peaks': batch['n_peaks'],
            'max_n_peaks': batch['max_n_peaks'],
            'motif_mean_std': batch['motif_mean_std'],
        }
        if perturb:
            result['output_bias'] = False
        return result

    def forward(self, sample_peak_sequence, chunk_size, n_peaks, max_n_peaks, motif_mean_std, output_bias=True):
        x = self.motif_scanner(sample_peak_sequence)
        atpm, aprofile = self.atac_attention(
            x, chunk_size, n_peaks, max_n_peaks)
        if not output_bias:
            logging.info(
                'output_bias is False, return atpm and aprofile for atac model only')
            return {'atpm': atpm, 'aprofile': aprofile}
        if self.with_bias:
            bias_output = self.bias_model(
                sample_peak_sequence, chunk_size, n_peaks, max_n_peaks, motif_mean_std)
            bias_atpm, bias_aprofile = bias_output['atpm'], bias_output['aprofile']
            atpm = torch.logsumexp(torch.stack(
                [atpm, bias_atpm], dim=0), dim=0)
            diff_length = aprofile.shape[1] - bias_aprofile.shape[1]
            crop_length = diff_length // 2
            bias_aprofile = F.pad(
                bias_aprofile, (crop_length, diff_length - crop_length), "constant", 0)
            aprofile = aprofile + bias_aprofile
        return {'atpm': atpm, 'aprofile': aprofile}

@dataclass
class GETNucleotideV1MotifAdaptor(BaseGETModelConfig):
    motif_scanner: MotifScannerConfig = field(
        default_factory=MotifScannerConfig)
    atac_attention: SplitPoolConfig = field(
        default_factory=SplitPoolConfig)
    region_embed: RegionEmbedConfig = field(default_factory=RegionEmbedConfig)


class GETNucleotideV1MotifAdaptor(BaseGETModel):
    def __init__(self, cfg: GETNucleotideV1MotifAdaptor):
        super().__init__(cfg)
        self.motif_scanner = MotifScanner(cfg.motif_scanner)
        conv_channel = cfg.motif_scanner.num_motif+2
        self.conv_blocks = nn.ModuleList([
            nn.Conv1d(conv_channel, 128, 3, padding=1),
            ConvBlock(128,
                      128),
        ])
        self.atac_attention = SplitPool(cfg.atac_attention)
        self.proj = nn.Linear(128,  # (B,R,M ->B,R,D)
                              cfg.region_embed.num_features)
        self.apply(self._init_weights)

    def get_input(self, batch):
        return {'sample_peak_sequence': batch['sample_peak_sequence'],
                'sample_track': batch['sample_track'],
                'chunk_size': batch['chunk_size'],
                'n_peaks': batch['n_peaks'],
                'max_n_peaks': batch['max_n_peaks']}

    def forward(self, sample_peak_sequence, sample_track, chunk_size, n_peaks, max_n_peaks):
        x = self.motif_scanner(sample_peak_sequence)
        x = x.permute(0, 2, 1)
        for conv in self.conv_blocks:
            x = conv(x)
        x = x.permute(0, 2, 1)
        # concat atac to x
        # x = torch.cat([x, sample_track.unsqueeze(-1)], dim=-1)
        x = self.atac_attention(
            x, chunk_size, n_peaks, max_n_peaks)
        # project D to 283
        x = self.proj(x)  # B, R, 283
        x = F.relu(x)
        return x

    def before_loss(self, output, batch):
        pred = {'motif': output[:,:,:-1]}
        obs = {'motif': batch['region_motif'][:,:,:-1]}
        return pred, obs

    def generate_dummy_data(self):
        B, R, L = 2, 1, 2000
        return {
            'sample_peak_sequence': torch.randint(0, 4, (B, R * L, 4)).float(),
        }



@dataclass
class GETNucleotideMotifAdaptor(BaseGETModelConfig):
    motif_scanner: MotifScannerConfig = field(
        default_factory=MotifScannerConfig)
    atac_attention: SplitPoolConfig = field(
        default_factory=SplitPoolConfig)
    region_embed: RegionEmbedConfig = field(default_factory=RegionEmbedConfig)


class GETNucleotideMotifAdaptor(BaseGETModel):
    def __init__(self, cfg: GETNucleotideMotifAdaptor):
        super().__init__(cfg)
        self.motif_scanner = MotifScanner(cfg.motif_scanner)
        conv_channel = cfg.motif_scanner.num_motif+2
        self.conv_blocks = nn.ModuleList([
            nn.Conv1d(conv_channel, 128, 3, padding=1),
            ConvBlock(128,
                      128),
        ])
        self.atac_attention = SplitPool(cfg.atac_attention)
        self.region_embed = RegionEmbed(cfg.region_embed)
        self.region_embed.eval()
        self.proj = nn.Linear(128,  # (B,R,M ->B,R,D)
                              cfg.region_embed.embed_dim)
        self.apply(self._init_weights)

    def get_input(self, batch):
        return {'sample_peak_sequence': batch['sample_peak_sequence'],
                'sample_track': batch['sample_track'],
                'chunk_size': batch['chunk_size'],
                'n_peaks': batch['n_peaks'],
                'max_n_peaks': batch['max_n_peaks']}

    def forward(self, sample_peak_sequence, sample_track, chunk_size, n_peaks, max_n_peaks):
        x = self.motif_scanner(sample_peak_sequence)
        x = x.permute(0, 2, 1)
        for conv in self.conv_blocks:
            x = conv(x)
        x = x.permute(0, 2, 1)
        # concat atac to x
        # x = torch.cat([x, sample_track.unsqueeze(-1)], dim=-1)
        x = self.atac_attention(
            x, chunk_size, n_peaks, max_n_peaks)
        # project D to 283
        x = self.proj(x)  # B, R, 283
        return x

    def before_loss(self, output, batch):
        pred = {'motif': output}
        obs = {'motif': self.region_embed(batch['region_motif']).detach()}
        return pred, obs

    def generate_dummy_data(self):
        B, R, L = 2, 1, 2000
        return {
            'sample_peak_sequence': torch.randint(0, 4, (B, R * L, 4)).float(),
        }


@dataclass
class GETNucleotideRegionFinetuneExpHiCAxialModelConfig(BaseGETModelConfig):
    motif_scanner: MotifScannerConfig = field(
        default_factory=MotifScannerConfig)
    atac_attention: SplitPoolConfig = field(default_factory=SplitPoolConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    head_exp: ExpressionHeadConfig = field(
        default_factory=ExpressionHeadConfig)
    header_hic: ContactMapHeadConfig = field(
        default_factory=ContactMapHeadConfig)
    header_abc: ContactMapHeadConfig = field(
        default_factory=ContactMapHeadConfig)
    distance_contact_map: DistanceContactHeadConfig = field(
        default_factory=DistanceContactHeadConfig)


class GETNucleotideRegionFinetuneExpHiCAxial(BaseGETModel):
    def __init__(self, cfg: GETNucleotideRegionFinetuneExpHiCAxialModelConfig):
        super().__init__(cfg)
        self.motif_scanner = MotifScanner(cfg.motif_scanner)
        self.conv_blocks = nn.ModuleList([
            ConvBlock(cfg.motif_scanner.num_motif,
                      cfg.motif_scanner.num_motif),
            ConvBlock(cfg.motif_scanner.num_motif,
                      cfg.motif_scanner.num_motif),
            ConvBlock(cfg.motif_scanner.num_motif,
                      cfg.motif_scanner.num_motif),
            ConvBlock(cfg.motif_scanner.num_motif, cfg.motif_scanner.num_motif)
        ])
        self.atac_attention = SplitPool(cfg.atac_attention)
        self.encoder = GETTransformerWithContactMapAxial(**cfg.encoder)
        self.head_exp = ExpressionHead(cfg.head_exp)
        self.distance_contact_map = DistanceContactHead(
            cfg.distance_contact_map)
        self.distance_contact_map.eval()
        self.proj_distance = nn.Linear(cfg.embed_dim, 1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.proj = nn.Linear(cfg.motif_scanner.num_motif, cfg.embed_dim)
        logging.info(
            f"GETNucleotideRegionFinetuneExpHiCAxial can only be used with quantitative_atac=True in order to generate the ABC score.")
        self.apply(self._init_weights)

    def get_input(self, batch, perturb=False):
        peak_coord = batch['peak_coord']
        peak_coord_mean = peak_coord[:, :, 0]
        peak_coord_mean_col = peak_coord_mean.unsqueeze(2)
        peak_coord_mean_row = peak_coord_mean.unsqueeze(1)
        distance = torch.log10(
            (peak_coord_mean_col - peak_coord_mean_row).abs() + 1).unsqueeze(1)

        sample_peak_sequence = batch['sample_peak_sequence']
        sample_track = batch['sample_track']
        chunk_size = batch['chunk_size']
        n_peaks = batch['n_peaks']
        max_n_peaks = batch['max_n_peaks']

        batch['distance_map'] = distance
        return {
            'sample_peak_sequence': sample_peak_sequence,
            'sample_track': sample_track,
            'chunk_size': chunk_size,
            'n_peaks': n_peaks,
            'max_n_peaks': max_n_peaks,
            'distance_map': distance,
        }

    def forward(self, sample_peak_sequence, sample_track, chunk_size, n_peaks, max_n_peaks, distance_map):
        x = self.motif_scanner(sample_peak_sequence)
        x = x.permute(0, 2, 1)
        for conv in self.conv_blocks:
            x = conv(x)
        x = x.permute(0, 2, 1)
        x = self.atac_attention(x, chunk_size, n_peaks, max_n_peaks)
        x = self.proj(x)

        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x, fused_distance_map, _ = self.encoder(x, distance_map)
        x = x[:, 1:]
        exp = nn.Softplus()(self.head_exp(x))
        fused_distance_map = self.proj_distance(fused_distance_map)
        return exp, fused_distance_map.squeeze(3)

    def before_loss(self, output, batch):
        exp, hic_contact_map = output
        atac = batch['sample_track'][:, :, -1]
        atac = torch.sqrt(atac.unsqueeze(1) * atac.unsqueeze(2))

        if len(batch['hic_matrix'][0].shape) >= 2:
            hic = batch['hic_matrix'].float()
            real_hic = True
        else:
            logging.info(
                f"batch['hic_matrix'] is not a matrix, using the distance contact map instead.")
            hic = self.distance_contact_map(
                batch['distance_map']).detach().squeeze(1)
            real_hic = False

        pred = {
            'exp': exp,
            'hic': hic_contact_map,
        }
        obs = {
            'exp': batch['exp_label'],
            'hic': hic,
        }
        if real_hic:
            mask = hic.sum(1).sum(1) == 0
            obs['hic'][mask] = pred['hic'][mask].detach().float()
        return pred, obs

    def generate_dummy_data(self):
        B, R, L = 2, 1, 2000
        return {
            'sample_peak_sequence': torch.randint(0, 4, (B, R * L, 4)).float(),
            'sample_track': torch.randn(B, R, 283).float(),
            'chunk_size': torch.tensor([R]),
            'n_peaks': torch.tensor([R]),
            'max_n_peaks': torch.tensor([R]),
            'peak_coord': torch.randn(B, R, 1).float(),
            'distance_map': torch.randn(B, R, R).float(),
        }



@dataclass
class GETNucleotideRegionFinetuneExpConfig(BaseGETModelConfig):
    motif_scanner: MotifScannerConfig = field(
        default_factory=MotifScannerConfig)
    atac_attention: SplitPoolConfig = field(default_factory=SplitPoolConfig)
    region_embed: RegionEmbedConfig = field(default_factory=RegionEmbedConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    head_exp: ExpressionHeadConfig = field(
        default_factory=ExpressionHeadConfig)


class GETNucleotideRegionFinetuneExp(BaseGETModel):
    def __init__(self, cfg: GETNucleotideRegionFinetuneExpConfig):
        super().__init__(cfg)
        self.motif_scanner = MotifScanner(cfg.motif_scanner)
        conv_channel = cfg.motif_scanner.num_motif+2
        self.conv_blocks = nn.ModuleList([
            nn.Conv1d(conv_channel, 128, 3, padding=1),
            ConvBlock(128,
                      128),
        ])
        self.atac_attention = SplitPool(cfg.atac_attention)
        self.proj = nn.Linear(128,  # (B,R,M ->B,R,D)
                              cfg.region_embed.num_features)
        self.region_embed = RegionEmbed(cfg.region_embed)
        self.encoder = GETTransformer(**cfg.encoder)
        self.head_exp = ExpressionHead(cfg.head_exp)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.apply(self._init_weights)

    def get_input(self, batch, perturb=False):

        sample_peak_sequence = batch['sample_peak_sequence']
        sample_track = batch['sample_track']
        chunk_size = batch['chunk_size']
        n_peaks = batch['n_peaks']
        max_n_peaks = batch['max_n_peaks']

        return {
            'sample_peak_sequence': sample_peak_sequence,
            'sample_track': sample_track,
            'chunk_size': chunk_size,
            'n_peaks': n_peaks,
            'max_n_peaks': max_n_peaks,
        }

    def forward(self, sample_peak_sequence, sample_track, chunk_size, n_peaks, max_n_peaks):
        x = self.motif_scanner(sample_peak_sequence)
        x = x.permute(0, 2, 1)
        for conv in self.conv_blocks:
            x = conv(x)
        x = x.permute(0, 2, 1)
        x = self.atac_attention(x, chunk_size, n_peaks, max_n_peaks)
        x = self.proj(x)
        x = F.relu(x).detach()
        x[:,:,282]=1
        x = self.region_embed(x)


        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x, _ = self.encoder(x)
        x = x[:, 1:]
        exp = nn.Softplus()(self.head_exp(x))
        return exp

    def before_loss(self, output, batch):
        pred = {
            'exp': output,
        }
        obs = {
            'exp': batch['exp_label'],
        }
        return pred, obs

    def generate_dummy_data(self):
        B, R, L = 2, 1, 2000
        return {
            'sample_peak_sequence': torch.randint(0, 4, (B, R * L, 4)).float(),
            'sample_track': torch.randn(B, R, 283).float(),
            'chunk_size': torch.tensor([R]),
            'n_peaks': torch.tensor([R]),
            'max_n_peaks': torch.tensor([R]),
            'peak_coord': torch.randn(B, R, 1).float(),
            'distance_map': torch.randn(B, R, R).float(),
        }



@dataclass
class GETNucleotideRegionFinetuneATACConfig(BaseGETModelConfig):
    motif_scanner: MotifScannerConfig = field(
        default_factory=MotifScannerConfig)
    atac_attention: SplitPoolConfig = field(default_factory=SplitPoolConfig)
    region_embed: RegionEmbedConfig = field(default_factory=RegionEmbedConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    head_exp: ExpressionHeadConfig = field(
        default_factory=ExpressionHeadConfig)


class GETNucleotideRegionFinetuneATAC(BaseGETModel):
    def __init__(self, cfg: GETNucleotideRegionFinetuneATACConfig):
        super().__init__(cfg)
        self.motif_scanner = MotifScanner(cfg.motif_scanner)
        conv_channel = cfg.motif_scanner.num_motif+2
        self.conv_blocks = nn.ModuleList([
            nn.Conv1d(conv_channel, 128, 3, padding=1),
            ConvBlock(128,
                      128),
        ])
        self.atac_attention = SplitPool(cfg.atac_attention)
        self.proj = nn.Linear(128,  # (B,R,M ->B,R,D)
                              cfg.region_embed.embed_dim)
        self.encoder = GETTransformer(**cfg.encoder)
        self.head_exp = ExpressionHead(cfg.head_exp)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.apply(self._init_weights)

    def get_input(self, batch, perturb=False):

        sample_peak_sequence = batch['sample_peak_sequence']
        sample_track = batch['sample_track']
        chunk_size = batch['chunk_size']
        n_peaks = batch['n_peaks']
        max_n_peaks = batch['max_n_peaks']

        return {
            'sample_peak_sequence': sample_peak_sequence,
            'sample_track': sample_track,
            'chunk_size': chunk_size,
            'n_peaks': n_peaks,
            'max_n_peaks': max_n_peaks,
        }

    def forward(self, sample_peak_sequence, sample_track, chunk_size, n_peaks, max_n_peaks):
        x = self.motif_scanner(sample_peak_sequence)
        x = x.permute(0, 2, 1)
        for conv in self.conv_blocks:
            x = conv(x)
        x = x.permute(0, 2, 1)
        x = self.atac_attention(x, chunk_size, n_peaks, max_n_peaks)
        x = self.proj(x)


        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x, _ = self.encoder(x)
        x = x[:, 1:]
        exp = nn.Softplus()(self.head_exp(x))
        return exp

    def before_loss(self, output, batch):
        pred = {
            'atpm': output.squeeze(2),
        }
        if 'region_motif' in batch:
            obs = {
                    'atpm': batch['region_motif'][:, :, -1],
                }
        else:
            obs = {
                'atpm': batch['atpm'],
            }
        return pred, obs

    def generate_dummy_data(self):
        B, R, L = 2, 1, 2000
        return {
            'sample_peak_sequence': torch.randint(0, 4, (B, R * L, 4)).float(),
            'sample_track': torch.randn(B, R, 283).float(),
            'chunk_size': torch.tensor([R]),
            'n_peaks': torch.tensor([R]),
            'max_n_peaks': torch.tensor([R]),
            'peak_coord': torch.randn(B, R, 1).float(),
            'distance_map': torch.randn(B, R, R).float(),
        }



@dataclass
class GETNucleotideRegionFinetuneExpHiCABCConfig(BaseGETModelConfig):
    motif_scanner: MotifScannerConfig = field(
        default_factory=MotifScannerConfig)
    atac_attention: SplitPoolConfig = field(default_factory=SplitPoolConfig)
    region_embed: RegionEmbedConfig = field(default_factory=RegionEmbedConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    head_exp: ExpressionHeadConfig = field(
        default_factory=ExpressionHeadConfig)
    header_hic: ContactMapHeadConfig = field(
        default_factory=ContactMapHeadConfig)
    header_abc: ContactMapHeadConfig = field(
        default_factory=ContactMapHeadConfig)
    distance_contact_map: DistanceContactHeadConfig = field(
        default_factory=DistanceContactHeadConfig)


class GETNucleotideRegionFinetuneExpHiCABC(BaseGETModel):
    def __init__(self, cfg: GETNucleotideRegionFinetuneExpHiCABCConfig):
        super().__init__(cfg)
        self.motif_scanner = MotifScanner(cfg.motif_scanner)
        self.conv_blocks = nn.ModuleList([
            ConvBlock(cfg.motif_scanner.num_motif,
                      cfg.motif_scanner.num_motif),
            ConvBlock(cfg.motif_scanner.num_motif,
                      cfg.motif_scanner.num_motif),
            ConvBlock(cfg.motif_scanner.num_motif,
                      cfg.motif_scanner.num_motif),
            ConvBlock(cfg.motif_scanner.num_motif, cfg.motif_scanner.num_motif)
        ])
        self.atac_attention = SplitPool(cfg.atac_attention)
        self.proj = nn.Linear(cfg.motif_scanner.num_motif, cfg.embed_dim)
        self.region_embed = RegionEmbed(cfg.region_embed)
        self.region_embed.eval()
        self.encoder = GETTransformerWithContactMap(**cfg.encoder)
        self.head_exp = ExpressionHead(cfg.head_exp)
        self.hic_header = ContactMapHead(cfg.header_hic)
        self.abc_header = ContactMapHead(cfg.header_abc)
        self.distance_contact_map = DistanceContactHead(
            cfg.distance_contact_map)
        self.distance_contact_map.eval()
        self.proj_distance = nn.Linear(cfg.embed_dim, 128)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        logging.info(
            f"GETNucleotideRegionFinetuneExpHiCABC can only be used with quantitative_atac=True in order to generate the ABC score.")
        self.apply(self._init_weights)

    def get_input(self, batch, perturb=False):
        peak_coord = batch['peak_coord']
        peak_coord_mean = peak_coord[:, :, 0]
        peak_coord_mean_col = peak_coord_mean.unsqueeze(2)
        peak_coord_mean_row = peak_coord_mean.unsqueeze(1)
        distance = torch.log10(
            (peak_coord_mean_col - peak_coord_mean_row).abs() + 1).unsqueeze(1)

        sample_peak_sequence = batch['sample_peak_sequence']
        sample_track = batch['sample_track']
        chunk_size = batch['chunk_size']
        n_peaks = batch['n_peaks']
        max_n_peaks = batch['max_n_peaks']

        batch['distance_map'] = distance
        return {
            'sample_peak_sequence': sample_peak_sequence,
            'sample_track': sample_track,
            'chunk_size': chunk_size,
            'n_peaks': n_peaks,
            'max_n_peaks': max_n_peaks,
            'distance_map': distance,
        }

    def forward(self, sample_peak_sequence, sample_track, chunk_size, n_peaks, max_n_peaks, distance_map):
        x = self.motif_scanner(sample_peak_sequence)
        x = x.permute(0, 2, 1)
        for conv in self.conv_blocks:
            x = conv(x)
        x = x.permute(0, 2, 1)
        x = self.atac_attention(x, chunk_size, n_peaks, max_n_peaks)
        x = self.proj(x)
        motif = x.clone()


        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x, fused_distance_map, _ = self.encoder(x, distance_map)
        x = x[:, 1:]
        exp = nn.Softplus()(self.head_exp(x))
        fused_distance_map = self.proj_distance(
            fused_distance_map).transpose(1, 3).transpose(2, 3)
        hic = self.hic_header(fused_distance_map).squeeze(1)
        abc = self.abc_header(fused_distance_map).squeeze(1)
        return exp, hic, abc, motif

    def before_loss(self, output, batch):
        exp, hic_contact_map, abc_contact_map, motif = output
        # atac = batch['sample_track'][:, :, -1]
        # atac = torch.sqrt(atac.unsqueeze(1) * atac.unsqueeze(2))

        if len(batch['hic_matrix'][0].shape) >= 2:
            hic = batch['hic_matrix'].float()
            real_hic = True
        else:
            logging.info(
                f"batch['hic_matrix'] is not a matrix, using the distance contact map instead.")
            hic = self.distance_contact_map(
                batch['distance_map']).detach().squeeze(1)
            real_hic = False
        # abc = atac * hic
        pred = {
            'exp': exp,
            'hic': hic_contact_map,
            'motif': motif,
            # 'abc': abc_contact_map,
        }
        obs = {
            'exp': batch['exp_label'],
            'hic': hic,
            'motif': self.region_embed(batch['region_motif']).detach()
            # 'abc': abc,
        }
        if real_hic:
            mask = hic.sum(1).sum(1) == 0
            obs['hic'][mask] = pred['hic'][mask].detach()
        return pred, obs

    def generate_dummy_data(self):
        B, R, L = 2, 1, 2000
        return {
            'sample_peak_sequence': torch.randint(0, 4, (B, R * L, 4)).float(),
            'sample_track': torch.randn(B, R, 283).float(),
            'chunk_size': torch.tensor([R]),
            'n_peaks': torch.tensor([R]),
            'max_n_peaks': torch.tensor([R]),
            'peak_coord': torch.randn(B, R, 1).float(),
            'distance_map': torch.randn(B, R, R).float(),
        }
