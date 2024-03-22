from dataclasses import dataclass, field
from unittest.mock import Base

import hydra
import lightning as L
from matplotlib.pyplot import cla
import torch
import torch.nn as nn
import torch.utils.data
import torchmetrics
from caesar.io.zarr_io import DenseZarrIO
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins import MixedPrecision
from lightning.pytorch.utilities import grad_norm
from omegaconf import MISSING, DictConfig
from torch.nn.init import trunc_normal_

from get_model.dataset.collate import get_rev_collate_fn
from get_model.dataset.dataset import build_dataset_zarr
from get_model.model.modules import (BaseModule, BaseConfig, ATACSplitPool, ATACSplitPoolConfig,
                                     ATACSplitPoolMaxNorm,
                                     ATACSplitPoolMaxNormConfig, ConvPool,
                                     ConvPoolConfig, MotifScanner,
                                     MotifScannerConfig, RegionEmbed,
                                     RegionEmbedConfig, SplitPool,
                                     SplitPoolConfig)
from get_model.model.transformer import GETTransformer
from get_model.optim import create_optimizer
from get_model.utils import cosine_scheduler, load_checkpoint, remove_keys


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

@dataclass
class GETPretrainConfig:
    _target_: str = "get_model.model.model_refactored.GETPretrain"
    motif_scanner: MotifScannerConfig = MISSING
    atac_attention: ATACSplitPoolConfig = MISSING
    region_embed: RegionEmbedConfig = MISSING
    encoder: EncoderConfig = MISSING

@dataclass
class GETPretrainMaxNormConfig(GETPretrainConfig):
    _target_: str = "get_model.model.model_refactored.GETPretrainMaxNorm"
    atac_attention: ATACSplitPoolMaxNormConfig = MISSING

@dataclass
class LossConfig:
    components: dict = MISSING
    weights: dict = MISSING

@dataclass
class MetricsConfig:
    masked = MISSING

@dataclass
class DatasetConfig:
    data_set: str = "Expression_Finetune_Fetal"
    eval_data_set: str = "Expression_Finetune_Fetal.fetal_eval"
    data_path: str = "/pmglocal/xf2217/get_data/"
    batch_size: int = 16
    num_workers: int = 16
    n_peaks_lower_bound: int = 5
    n_peaks_upper_bound: int = 10
    max_peak_length: int = 5000
    center_expand_target: int = 500
    use_insulation: bool = False
    preload_count: int = 10
    random_shift_peak: int = 10
    pin_mem: bool = True
    peak_name: str = "peaks_q0.01_tissue_open_exp"
    negative_peak_name: str|None = None
    n_packs: int = 1
    leave_out_celltypes: str = "Astrocyte"
    leave_out_chromosomes: str = "chr4,chr14"
    dataset_size: int = 4096
    additional_peak_columns: list = field(default_factory=lambda: ['Expression_positive', 'Expression_negative', 'aTPM', 'TSS'])
    padding: int = 0
    mask_ratio: float = 0.5
    insulation_subsample_ratio: int = 1
    negative_peak_ratio: int = 0
    peak_inactivation: str|None = None
    non_redundant: bool = False
    filter_by_min_depth: bool = False
    hic_path: str|None = None

@dataclass
class OptimizerConfig:
    lr: float = 0.001
    min_lr: float = 0.0001
    weight_decay: float = 0.05
    opt: str = 'adamw'
    opt_eps: float = 1e-8
    opt_betas: list = field(default_factory=lambda: [0.9, 0.95])

@dataclass
class TrainingConfig:
    num_devices: int = 1
    save_ckpt_freq: int = 10
    epochs: int = 100
    warmup_epochs: int = 5
    accumulate_grad_batches: int = 1
    clip_grad: float|None = None
    use_fp16: bool = True
    output_dir: str = "/pmglocal/xf2217/output"
    optimizer: OptimizerConfig = MISSING

@dataclass
class WandbConfig:
    project_name: str = "pretrain"
    run_name: str = "experiment_1"

@dataclass
class FinetuneConfig:
    checkpoint: str|None = None
    model_prefix: str = "model."
    patterns_to_freeze: list = field(default_factory=lambda: [
                                    "motif_scanner"])

@dataclass
class Config:
    model: GETPretrainMaxNormConfig = MISSING
    loss: LossConfig = MISSING
    metrics: MetricsConfig = MISSING
    dataset: DatasetConfig = MISSING
    training: TrainingConfig = MISSING
    wandb: WandbConfig = MISSING
    finetune: FinetuneConfig = MISSING

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="model", name="base_model", node=GETPretrainMaxNormConfig)
cs.store(group="model.motif_scanner", name="base_motif_scanner", node=MotifScannerConfig)
cs.store(group="model.atac_attention", name="base_atac_attention", node=ATACSplitPoolMaxNormConfig)
cs.store(group="model.region_embed", name="base_region_embed", node=RegionEmbedConfig)
cs.store(group="model.encoder", name="base_encoder", node=EncoderConfig)
cs.store(group="loss", name="base_loss", node=LossConfig)
cs.store(group="metrics", name="base_metrics", node=MetricsConfig)
cs.store(group="dataset", name="base_dataset", node=DatasetConfig)
cs.store(group="training", name="base_training", node=TrainingConfig)
cs.store(group="training.optimizer", name="base_optimizer", node=OptimizerConfig)
cs.store(group="wandb", name="base_wandb", node=WandbConfig)
cs.store(group="finetune", name="base_finetune", node=FinetuneConfig)

class GETLoss(nn.Module):
    def __init__(self, cfg):
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
    def __init__(self, _cfg_):
        super(RegressionMetrics, self).__init__()
        self.cfg = _cfg_
        self.metrics = nn.ModuleDict({
            target: nn.ModuleDict({
                metric_name: self._get_metric(metric_name) for metric_name in metric_names
            }) for target, metric_names in _cfg_.items()
        })

    def _get_metric(self, metric_name):
        if metric_name == 'pearson':
            return torchmetrics.PearsonCorrCoef()
        elif metric_name == 'spearman':
            return torchmetrics.SpearmanCorrCoef()
        elif metric_name == 'mse':
            return torchmetrics.MeanSquaredError()
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")

    def forward(self, _pred_, _obs_):
        """Compute the metrics"""
        batch_size = _pred_[list(_pred_.keys())[0]].shape[0]
        result = {
            target: {
                metric_name: metric(
                    _pred_[target][:, 0, :].reshape(-1, 1),
                    _obs_[target][:, 0, :].reshape(-1, 1))
                for metric_name, metric in target_metrics.items()
            }
            for target, target_metrics in self.metrics.items()
        }
        # flatten the result
        result = {f"{target}_{metric_name}": result[target][metric_name]
                  for target in result for metric_name in result[target]}
        return result


class BaseGETModel(BaseModule):
    def __init__(self, cfg: BaseConfig):
        super(BaseGETModel, self).__init__()
        self.cfg = cfg

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

    def before_loss(self, output, target):
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
    

class GETPretrain(BaseGETModel):
    def __init__(self, cfg: GETPretrainConfig):
        super().__init__(GETPretrain)
        self.motif_scanner = MotifScanner(**cfg.motif_scanner)
        self.atac_attention = ATACSplitPool(**cfg.atac_attention)
        self.split_pool = SplitPool()
        self.region_embed = RegionEmbed(**cfg.region_embed)
        self.encoder = GETTransformer(**cfg.encoder)
        self.head_mask = nn.Linear(**cfg.head_mask)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.mask_token.embed_dim))
        trunc_normal_(self.mask_token, std=cfg.mask_token.std)
        self.loss = GETLoss(cfg.loss)
        self.metrics = RegressionMetrics(cfg.metrics)

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
        x_region = self.split_pool(x, chunk_size, n_peaks, max_n_peaks)
        x_original = self.atac_attention(
            x, x_region, sample_track, chunk_size, n_peaks, max_n_peaks)
        x = self.region_embed(x_original)
        B, N, C = x_original.shape
        mask_token = self.mask_token.expand(B, N, -1)
        w = loss_mask.type_as(mask_token)
        x = x * (1 - w) + mask_token * w
        x, _ = self.encoder(x, mask=padding_mask)
        x_masked = self.head_mask(x)
        return x_masked, x_original, loss_mask

    def before_loss(self, output, target=None):
        x_masked, x_original, loss_mask = output
        pred = {'masked': x_masked * loss_mask}
        obs = {'masked': x_original * loss_mask}
        return pred, obs
    
class GETPretrainMaxNorm(GETPretrain):
    def __init__(self, cfg: GETPretrainMaxNormConfig):
        super().__init__(cfg)
        self.atac_attention = ATACSplitPoolMaxNorm(**cfg.atac_attention)


class LitModel(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = self.get_model()
        self.loss = self.model.loss
        self.metrics = self.model.metrics
        self.save_hyperparameters()

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def get_model(self):
        model = instantiate(self.cfg.model)
        if self.cfg.finetune.checkpoint is not None:
            checkpoint_model = load_checkpoint(
                self.cfg.finetune.checkpoint)
            state_dict = model.state_dict()
            remove_keys(checkpoint_model, state_dict)
            # checkpoint_model = rename_keys(checkpoint_model)
            model.load_state_dict(checkpoint_model, strict=False)
        model.freeze_layers(
            patterns_to_freeze=self.cfg.finetune.patterns_to_freeze, invert_match=False)
        return model

    def forward(self, batch):
        return self.model(**batch)

    def _shared_step(self, batch, batch_idx, stage='train'):
        batch = self.model.get_input(batch)
        output = self(batch)
        pred, obs = self.model.before_loss(output)
        loss = self.loss(pred, obs)
        # if loss is a dict, rename the keys with the stage prefix
        if isinstance(loss, dict):
            loss = {f"{stage}_{key}": value for key,
                    value in loss.items()}
            self.log_dict(loss, batch_size=self.cfg.dataset.batch_size)
        loss = self.model.after_loss(loss)
        return loss, pred, obs

    def training_step(self, batch, batch_idx):
        loss, pred, obs = self._shared_step(batch, batch_idx, stage='train')
        self.log("train_loss", loss, batch_size=self.cfg.dataset.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, obs = self._shared_step(batch, batch_idx, stage='val')
        metrics = self.metrics(pred, obs)
        self.log_dict(metrics, batch_size=self.cfg.dataset.batch_size)
        self.log("val_loss", loss, batch_size=self.cfg.dataset.batch_size)

    def configure_optimizers(self):
        optimizer = create_optimizer(self.cfg.optimizer, self.model)
        num_training_steps_per_epoch = (
            self.cfg.dataset.dataset_size // self.cfg.dataset.batch_size // self.cfg.training.num_devices
        )
        schedule = cosine_scheduler(
            base_value=self.cfg.optimizer.lr,
            final_value=self.cfg.optimizer.min_lr,
            epochs=self.cfg.training.epochs,
            niter_per_ep=num_training_steps_per_epoch,
            warmup_epochs=self.cfg.training.warmup_epochs,
            start_warmup_value=0,
            warmup_steps=-1
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: schedule[step],
        )
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        sequence_obj = DenseZarrIO(
            f'{self.cfg.dataset.data_path}/hg38.zarr', dtype='int8')
        sequence_obj.load_to_memory_dense()
        dataset_train = build_dataset_zarr(
            is_train=True, args=self.cfg.dataset, sequence_obj=sequence_obj)
        return torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            drop_last=True,
            collate_fn=get_rev_collate_fn,
        )

    def val_dataloader(self):
        sequence_obj = DenseZarrIO(
            f'{self.cfg.dataset.data_path}/hg38.zarr', dtype='int8')
        sequence_obj.load_to_memory_dense()
        dataset_eval = build_dataset_zarr(
            is_train=False, args=self.cfg.dataset, sequence_obj=sequence_obj)
        return torch.utils.data.DataLoader(
            dataset_eval,
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            drop_last=True,
            collate_fn=get_rev_collate_fn
        )


class GETDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        sequence_obj = DenseZarrIO(
            f'{self.cfg.dataset.data_path}/hg38.zarr', dtype='int8')
        sequence_obj.load_to_memory_dense()
        self.dataset_train = build_dataset_zarr(
            is_train=True, args=self.cfg.dataset, sequence_obj=sequence_obj)
        self.dataset_eval = build_dataset_zarr(
            is_train=False, args=self.cfg.dataset, sequence_obj=sequence_obj)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            drop_last=True,
            collate_fn=get_rev_collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_eval,
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            drop_last=True,
            collate_fn=get_rev_collate_fn,
        )


@hydra.main(config_path="../config/model/pretrain", config_name="template", version_base="1.3")
def main(cfg: Config):
    torch.set_float32_matmul_precision('medium')
    model = LitModel(cfg)
    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="gpu",
        num_sanity_val_steps=10,
        strategy="auto",
        profiler="simple",
        devices=cfg.training.num_devices,
        logger=[WandbLogger(project=cfg.wandb.project_name,
                            name=cfg.wandb.run_name)],
        callbacks=[ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, save_last=True, filename="best"),
                   LearningRateMonitor(logging_interval='step')],
        plugins=[MixedPrecision(precision='16-mixed', device="cuda")],
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        gradient_clip_val=cfg.training.clip_grad,
        log_every_n_steps=100,
        deterministic=True,
        default_root_dir=cfg.training.output_dir,
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
