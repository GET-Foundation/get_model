from dataclasses import dataclass

import hydra
import lightning as L
import torch
import torch.nn.functional as F
import torch.utils.data
from caesar.io.zarr_io import DenseZarrIO
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins import MixedPrecision
from lightning.pytorch.utilities import grad_norm
from omegaconf import MISSING, DictConfig

from get_model.config.config import (DatasetConfig, FinetuneConfig, LossConfig,
                                     MetricsConfig, TrainingConfig, OptimizerConfig,
                                     WandbConfig)

from get_model.dataset.collate import get_rev_collate_fn
from get_model.dataset.dataset import build_dataset_zarr
from get_model.model.modules import *
from get_model.model.model_refactored import *
from get_model.optim import create_optimizer
from get_model.utils import cosine_scheduler, load_checkpoint, remove_keys

@dataclass
class Config:
    model: BaseConfig = MISSING
    loss: LossConfig = MISSING
    metrics: MetricsConfig = MISSING
    dataset: DatasetConfig = MISSING
    training: TrainingConfig = MISSING
    wandb: WandbConfig = MISSING
    finetune: FinetuneConfig = MISSING

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

cs.store(group="loss", name="base_loss", node=LossConfig)
cs.store(group="metrics", name="base_metrics", node=MetricsConfig)
cs.store(group="dataset", name="base_dataset", node=DatasetConfig)
cs.store(group="training", name="base_training", node=TrainingConfig)
cs.store(group="training.optimizer", name="base_optimizer", node=OptimizerConfig)
cs.store(group="wandb", name="base_wandb", node=WandbConfig)
cs.store(group="finetune", name="base_finetune", node=FinetuneConfig)


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
        pred, obs = self.model.before_loss(output, batch)
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
    
# cs.store(group="model", name="base_finetune_model", node=GETFinetuneConfig)
# cs.store(group="model.head_exp", name="base_expression_head", node=ExpressionHeadConfig)
# cs.store(group="model", name="base_finetune_chrombpnet_bias_model", node=GETFinetuneChrombpNetBiasConfig)

def run(cfg: DictConfig):
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

@hydra.main(config_path="config/model/pretrain", config_name="GETPretrain", version_base="1.3")
def get_pretrain(cfg: DictConfig):

    run(cfg)

@hydra.main(config_path="config", config_name="pretrain_pc", version_base="1.3")
def get_pretrain_maxnorm(cfg: DictConfig):
    cs.store(group="model", name="base_model", node=GETPretrainMaxNormConfig)
    cs.store(group="model.motif_scanner", name="base_motif_scanner", node=MotifScannerConfig)
    cs.store(group="model.atac_attention", name="base_atac_attention", node=ATACSplitPoolMaxNormConfig)
    cs.store(group="model.region_embed", name="base_region_embed", node=RegionEmbedConfig)
    cs.store(group="model.encoder", name="base_encoder", node=EncoderConfig)
    run(cfg)

@hydra.main(config_path="config/model/finetune", config_name="GETFinetune", version_base="1.3")
def get_finetune(cfg: DictConfig):
    run(cfg)

@hydra.main(config_path="config/model/finetune", config_name="GETFinetuneMaxNorm", version_base="1.3")
def get_finetune_maxnorm(cfg: DictConfig):
    run(cfg)

@hydra.main(config_path="config/model/finetune", config_name="GETFinetuneChrombpNetBias", version_base="1.3")
def get_finetune_chrombpnet_bias(cfg: DictConfig):
    run(cfg)




if __name__ == "__main__":
    get_pretrain_maxnorm()
