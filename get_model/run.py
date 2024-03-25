import logging
import os.path

import hydra
import lightning as L
import pandas as pd
import torch
import torch.utils.data
from caesar.io.zarr_io import DenseZarrIO
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins import MixedPrecision
from lightning.pytorch.utilities import grad_norm
from omegaconf import MISSING, DictConfig

from get_model.config.config import *
from get_model.dataset.collate import get_rev_collate_fn
from get_model.dataset.zarr_dataset import DenseZarrIO, PretrainDataset
from get_model.model.model_refactored import *
from get_model.model.modules import *
from get_model.optim import create_optimizer
from get_model.utils import cosine_scheduler, load_checkpoint, remove_keys



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
        input = self.model.get_input(batch)
        output = self(input)
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

    def task_step(self, batch, batch_idx):
        loss, pred, obs = self._shared_step(batch, batch_idx, stage='predict')
        return pred, obs
    
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

    # def on_validation_end(self):
    #     # Perform inference on the mutations
    #     predictions = []
    #     for batch in self.inference_dataset:
    #         batch_predictions = self.model(batch)
    #         predictions.append(batch_predictions)

    #     # Run evaluation tasks
    #     for name, task in self.evaluation_tasks.items():
    #         task.predict(predictions)
    #         # log correlation


class GETDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def build_dataset_zarr(self, sequence_obj, is_train=True) -> None:
        config = self.cfg.dataset.dataset_configs[self.cfg.dataset_name]
        # merge config with self.cfg.dataset
        config = DictConfig({**self.cfg.dataset, **config})

        root = self.cfg.machine.data_path
        codebase = self.cfg.machine.codebase
        assembly = self.cfg.assembly
        dataset_size = config.dataset_size if is_train else config.eval_dataset_size
        zarr_dirs = [f'{root}/{zarr_dir}' for zarr_dir in config.zarr_dirs]
        if sequence_obj is None:
            sequence_obj = DenseZarrIO(f'{root}/{assembly}.zarr', dtype='int8')
            sequence_obj.load_to_memory_dense()
        else:
            logging.info('sequence_obj is provided')
            sequence_obj = sequence_obj

        # Create dataset with configuration parameters
        dataset = PretrainDataset(
            is_train=is_train,
            sequence_obj=sequence_obj,
            zarr_dirs=zarr_dirs,
            genome_seq_zarr=f'{root}/{assembly}.zarr',
            genome_motif_zarr=f'{root}/{assembly}_motif_result.zarr',
            use_insulation=config.use_insulation,
            insulation_paths=[
                f'{codebase}/data/{assembly}_4DN_average_insulation.ctcf.adjecent.feather',
                f'{codebase}/data/{assembly}_4DN_average_insulation.ctcf.longrange.feather'],
            hic_path=config.hic_path,

            peak_name=config.peak_name,
            additional_peak_columns=config.additional_peak_columns,
            max_peak_length=config.max_peak_length,
            center_expand_target=config.center_expand_target,
            n_peaks_lower_bound=config.n_peaks_lower_bound,
            n_peaks_upper_bound=config.n_peaks_upper_bound,
            n_peaks_sample_gap=config.n_peaks_upper_bound,
            non_redundant=config.non_redundant,
            filter_by_min_depth=config.filter_by_min_depth,

            preload_count=config.preload_count,
            n_packs=config.n_packs,

            padding=config.padding,
            mask_ratio=config.mask_ratio,
            negative_peak_name=config.negative_peak_name,
            negative_peak_ratio=config.negative_peak_ratio,
            random_shift_peak=config.random_shift_peak,
            peak_inactivation=config.peak_inactivation,

            leave_out_celltypes=config.leave_out_celltypes,
            leave_out_chromosomes=config.leave_out_chromosomes,
            dataset_size=dataset_size,
        )

        return dataset


    def prepare_data(self):
        pass

    def setup(self, stage=None):
        sequence_obj = DenseZarrIO(
            f'{self.cfg.data_path}/{self.cfg.assembly}.zarr', dtype='int8')
        sequence_obj.load_to_memory_dense()
        if stage == 'fit' or stage is None:
            self.dataset_train = self.build_dataset_zarr(sequence_obj=sequence_obj, is_train=True)
            self.dataset_val = self.build_dataset_zarr(sequence_obj=sequence_obj, is_train=False)
        if stage == 'test' or stage is None:
            self.dataset_test = self.build_dataset_zarr(sequence_obj=sequence_obj, is_train=False)
        if stage == 'predict' or stage is None:
            self.dataset_predict = self.build_dataset_zarr(sequence_obj=sequence_obj, is_train=False)
        if stage == 'validate' or stage is None:
            self.dataset_val = self.build_dataset_zarr(sequence_obj=sequence_obj, is_train=False)

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
            self.dataset_val,
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            drop_last=True,
            collate_fn=get_rev_collate_fn,
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            drop_last=True,
            collate_fn=get_rev_collate_fn,
        )
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_predict,
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            drop_last=True,
            collate_fn=get_rev_collate_fn,
        )
    

def run(cfg: DictConfig):
    torch.set_float32_matmul_precision('medium')
    model = LitModel(cfg)
    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="cpu",
        num_sanity_val_steps=10,
        strategy="auto",
        devices=cfg.machine.num_devices,
        logger=[WandbLogger(project=cfg.wandb.project_name,
                            name=cfg.wandb.run_name)],
        callbacks=[ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, save_last=True, filename="best"),
                   LearningRateMonitor(logging_interval='step')],
        # plugins=[MixedPrecision(precision='16-mixed', device="cpu")],
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        gradient_clip_val=cfg.training.clip_grad,
        log_every_n_steps=100,
        deterministic=True,
        default_root_dir=cfg.machine.output_dir,
    )

    trainer.fit(model, datamodule=GETDataModule(cfg))
