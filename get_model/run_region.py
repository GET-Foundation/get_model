import logging

import lightning as L
import pandas as pd
from sympy import Li
import torch
import torch.utils.data
from caesar.io.gencode import Gencode
from caesar.io.zarr_io import DenseZarrIO
from hydra.utils import instantiate
from lightning.pytorch.callbacks import (Callback, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins import MixedPrecision
from lightning.pytorch.utilities import grad_norm
from omegaconf import MISSING, DictConfig, OmegaConf
from sklearn import metrics
from tqdm import tqdm
import wandb
import seaborn as sns

from get_model.config.config import *
from get_model.dataset.collate import (get_perturb_collate_fn,
                                       get_rev_collate_fn)
from get_model.dataset.region_dataset import RegionDataset
from get_model.model.model_refactored import *
from get_model.model.modules import *
from get_model.optim import create_optimizer
from get_model.utils import cosine_scheduler, load_checkpoint, remove_keys
from run import LitModel
logging.disable(logging.WARN)


class RegionDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def build_training_dataset(self, is_train=True):
        return RegionDataset(**self.cfg.dataset, is_train=is_train)

    def build_inference_dataset(self, is_train=False):
        return RegionDataset(**self.cfg.dataset, is_train=is_train)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.dataset_train = self.build_training_dataset(is_train=True)
            self.dataset_val = self.build_training_dataset(is_train=False)
        if stage == 'predict':
            if self.cfg.task.test_mode == 'predict':
                self.dataset_predict = self.build_training_dataset(
                    is_train=False)
            elif self.cfg.task.test_mode == 'interpret':
                self.dataset_predict = self.build_inference_dataset()

        if stage == 'validate':
            self.dataset_val = self.build_training_dataset(is_train=False)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.cfg.machine.batch_size,
            num_workers=self.cfg.machine.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.cfg.machine.batch_size,
            num_workers=self.cfg.machine.num_workers,
            drop_last=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=self.cfg.machine.batch_size,
            num_workers=self.cfg.machine.num_workers,
            drop_last=True,
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_predict,
            batch_size=self.cfg.machine.batch_size,
            num_workers=self.cfg.machine.num_workers,
            drop_last=False,
        )


class RegionLitModel(LitModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def rename_keys(self, state_dict):
        """
        Rename the keys in the state dictionary.
        """
        new_state_dict = {}
        for key in state_dict.keys():
            new_key = key
            # Adjust keys according to the new model architecture
            new_key = new_key.replace("blocks.", "encoder.blocks.")
            new_key = new_key.replace("fc_norm.", "encoder.norm.")
            new_key = new_key.replace("head.", "head_exp.head.")
            new_key = new_key.replace(
                "region_embed.proj.", "region_embed.embed.")

            new_state_dict[new_key] = state_dict[key]
            # drop cls token
            if "cls" in new_key:
                del new_state_dict[new_key]
        # Adjust the weight dimensions if needed
        # Uncomment the next line if the weight dimension needs to be changed
        # new_state_dict['region_embed.embed.weight'] = new_state_dict['region_embed.embed.weight'].unsqueeze(2)
        return new_state_dict

    def validation_step(self, batch, batch_idx):
        loss, pred, obs = self._shared_step(batch, batch_idx, stage='val')
        print(pred['exp'].detach().cpu().numpy().flatten().max(),
              obs['exp'].detach().cpu().numpy().flatten().max())
        metrics = self.metrics(pred, obs)
        if batch_idx == 0:
            # log one example as scatter plot
            self.logger.experiment.log({
                "scatter": wandb.Image(sns.scatterplot(y=pred['exp'].detach().cpu().numpy().flatten(), x=obs['exp'].detach().cpu().numpy().flatten()))
            })
        self.log_dict(metrics, batch_size=self.cfg.machine.batch_size)
        self.log("val_loss", loss, batch_size=self.cfg.machine.batch_size)

    def get_model(self):
        model = instantiate(self.cfg.model)
        if self.cfg.finetune.checkpoint is not None:
            checkpoint_model = load_checkpoint(
                self.cfg.finetune.checkpoint)
            state_dict = model.state_dict()
            remove_keys(checkpoint_model, state_dict)
            if 'model' in checkpoint_model:
                checkpoint_model = checkpoint_model['model']
            checkpoint_model = self.rename_keys(checkpoint_model)
            model.load_state_dict(checkpoint_model, strict=True)
        model.freeze_layers(
            patterns_to_freeze=self.cfg.finetune.patterns_to_freeze, invert_match=False)
        return model

    def on_validation_epoch_end(self):
        pass


def run(cfg: DictConfig):
    model = RegionLitModel(cfg)
    print(OmegaConf.to_yaml(cfg))
    dm = RegionDataModule(cfg)
    model.dm = dm
    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="gpu",
        num_sanity_val_steps=0,
        strategy="auto",
        devices=cfg.machine.num_devices,
        logger=[WandbLogger(project=cfg.wandb.project_name,
                            name=cfg.wandb.run_name)],
        callbacks=[ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, save_last=True, filename="best"),
                   LearningRateMonitor(logging_interval='epoch')],
        # plugins=[MixedPrecision(precision='16-mixed', device="cuda")],
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        gradient_clip_val=cfg.training.clip_grad,
        log_every_n_steps=4,
        default_root_dir=cfg.machine.output_dir,
    )
    trainer.validate(model, datamodule=dm)
