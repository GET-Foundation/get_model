from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchmetrics
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig
from torch.nn.init import trunc_normal_

from get_model.model.modules import (ATACSplitPool, ATACSplitPoolConfig,
                                     ATACSplitPoolMaxNorm,
                                     ATACSplitPoolMaxNormConfig, BaseConfig,
                                     BaseModule, ConvPool, ConvPoolConfig,
                                     ExpressionHead, ExpressionHeadConfig,
                                     MotifScanner, MotifScannerConfig,
                                     RegionEmbed, RegionEmbedConfig, SplitPool,
                                     SplitPoolConfig, dict_to_device)
from get_model.model.transformer import GETTransformer


class MNLLLoss(nn.Module):
    def __init__(self):
        super(MNLLLoss, self).__init__()

    def forward(self, x, y):
        """Compute the loss use -y* log(softmax(x))"""
        return (-y.squeeze(1) * F.log_softmax(x, dim=-1)).mean()


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
                print(f"Freezed weights of {name}")

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
class GETFinetuneMaxNormModelConfig(GETFinetuneModelConfig):
    atac_attention: ATACSplitPoolMaxNormConfig = MISSING


class GETFinetuneMaxNorm(GETFinetune):
    def __init__(self, cfg: GETFinetuneMaxNormModelConfig):
        super().__init__(cfg)
        self.atac_attention = ATACSplitPoolMaxNorm(cfg.atac_attention)


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

        self.apply(self._init_weights)

    def get_input(self, batch):
        return {
            'region_motif': batch['region_motif'],
        }

    def forward(self, region_motif):

        x = self.region_embed(region_motif)
        x, _ = self.encoder(x)
        exp = nn.Softplus()(self.head_exp(x))
        return exp

    def before_loss(self, output, batch):
        tss_idx = batch['mask']
        pred = {'exp': output[tss_idx == 1]}
        obs = {'exp': batch['exp_label'][tss_idx == 1]}
        return pred, obs

    def generate_dummy_data(self):
        B, R, M = 2, 900, 283
        return {
            'region_motif': torch.randn(B, R, M).float().abs(),
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
        obs = {'atpm': batch['sample_track'].mean(dim=1).unsqueeze(-1),
               'aprofile': batch['sample_track']}
        pred['aprofile'], obs['aprofile'] = self.crop_output(
            pred['aprofile'], obs['aprofile'], B, R)
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

    def forward(self, sample_peak_sequence, chunk_size, n_peaks, max_n_peaks, motif_mean_std):
        x = self.motif_scanner(sample_peak_sequence)
        atpm, aprofile = self.atac_attention(
            x, chunk_size, n_peaks, max_n_peaks)
        if self.with_bias:
            bias_output = self.bias_model(
                sample_peak_sequence, chunk_size, n_peaks, max_n_peaks, motif_mean_std)
            bias_atpm, bias_aprofile = bias_output['atpm'], bias_output['aprofile']
            atpm = torch.logsumexp(torch.stack(
                [atpm, bias_atpm], dim=0), dim=0)
            # diff_length = aprofile.shape[1] - bias_aprofile.shape[1]
            # crop_length = diff_length // 2
            # bias_aprofile = F.pad(
            #     bias_aprofile, (crop_length, diff_length - crop_length), "constant", 0)
            aprofile = aprofile + bias_aprofile
        return {'atpm': atpm, 'aprofile': aprofile}
