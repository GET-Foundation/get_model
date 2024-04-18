import logging

import lightning as L
import seaborn as sns
import torch
import torch.utils.data
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.plugins import MixedPrecision
from omegaconf import MISSING, DictConfig, OmegaConf

import wandb
from get_model.config.config import *
from get_model.dataset.zarr_dataset import (ReferenceRegionDataset,
                                            ReferenceRegionMotif,
                                            ReferenceRegionMotifConfig)
from get_model.model.model_refactored import *
from get_model.model.modules import *
from get_model.optim import LayerDecayValueAssigner, create_optimizer
from get_model.run import GETDataModule, LitModel
from get_model.utils import (cosine_scheduler, load_checkpoint, remove_keys,
                             rename_lit_state_dict, rename_v1_finetune_keys,
                             rename_v1_pretrain_keys)


class ReferenceRegionDataModule(GETDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        logging.info("Init ReferenceRegionDataModule")
        cfg.dataset.reference_region_motif['root'] = self.cfg.machine.data_path
        self.reference_region_motif_cfg = ReferenceRegionMotifConfig(
            **cfg.dataset.reference_region_motif)
        self.reference_region_motif = ReferenceRegionMotif(
            self.reference_region_motif_cfg)
        print(self.reference_region_motif)

    def build_from_zarr_dataset(self, zarr_dataset):
        return ReferenceRegionDataset(self.reference_region_motif, zarr_dataset, quantitative_atac=self.cfg.dataset.quantitative_atac, sampling_step=self.cfg.dataset.sampling_step)

    def setup(self, stage=None):
        super().setup(stage)
        if stage == 'fit' or stage is None:
            self.dataset_train = self.build_from_zarr_dataset(
                self.dataset_train)
            self.dataset_val = self.build_from_zarr_dataset(self.dataset_val)
        if stage == 'predict':
            self.dataset_predict = self.build_from_zarr_dataset(
                self.dataset_predict)
        if stage == 'validate':
            self.dataset_val = self.build_from_zarr_dataset(self.dataset_val)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.cfg.machine.batch_size,
            num_workers=self.cfg.machine.num_workers,
            drop_last=True,
            shuffle=True,
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

    def validation_step(self, batch, batch_idx):
        loss, pred, obs = self._shared_step(batch, batch_idx, stage='val')
        # print(pred['exp'].detach().cpu().numpy().flatten().max(),
        #       obs['exp'].detach().cpu().numpy().flatten().max())

        if self.cfg.eval_tss:
            tss_idx = batch['mask']
            for key in pred:
                # pred[key] = (pred[key] * tss_idx)
                # obs[key] = (obs[key] * tss_idx)
                pred[key] = pred[key][tss_idx > 0].flatten()
                obs[key] = obs[key][tss_idx > 0].flatten()

        metrics = self.metrics(pred, obs)
        if batch_idx == 0 and self.cfg.log_image:
            # log one example as scatter plot
            for key in pred:
                self.logger.experiment.log({
                    f"scatter_{key}": wandb.Image(sns.scatterplot(y=pred[key].detach().cpu().numpy().flatten(), x=obs[key].detach().cpu().numpy().flatten()))
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
            strict = self.cfg.finetune.strict
            if 'model' in checkpoint_model:
                checkpoint_model = checkpoint_model['model']
            if 'state_dict' in checkpoint_model:
                checkpoint_model = checkpoint_model['state_dict']
                checkpoint_model = rename_lit_state_dict(checkpoint_model)
                model.load_state_dict(checkpoint_model, strict=strict)
            else:
                if self.cfg.finetune.pretrain_checkpoint:
                    checkpoint_model = rename_v1_pretrain_keys(
                        checkpoint_model)
                    model.load_state_dict(checkpoint_model, strict=strict)
                else:
                    checkpoint_model = rename_v1_finetune_keys(
                        checkpoint_model)
                    model.load_state_dict(checkpoint_model, strict=strict)
        model.freeze_layers(
            patterns_to_freeze=self.cfg.finetune.patterns_to_freeze, invert_match=False)
        print("Model = %s" % str(model))
        return model

    def on_validation_epoch_end(self):
        # save self.trainer.callback_metrics to a csv as one row
        metric_dict = dict_to_item(self.trainer.callback_metrics)
        metric_dict['count_filter'] = self.cfg.dataset.reference_region_motif.count_filter
        metric_dict['motif_scaler'] = self.cfg.dataset.reference_region_motif.motif_scaler
        metric_dict['quantitative_atac'] = self.cfg.quantitative_atac
        # as dataframe
        import pandas as pd
        df = pd.DataFrame(metric_dict, index=[0])
        # save to csv
        if not os.path.exists(f"{self.cfg.machine.output_dir}/val_metrics.csv"):
            df.to_csv(f"{self.cfg.machine.output_dir}/val_metrics.csv",
                      index=False)
        else:
            df.to_csv(f"{self.cfg.machine.output_dir}/val_metrics.csv",
                      mode='a', header=False, index=False)

    def configure_optimizers(self):
        num_layers = self.model.cfg.encoder.num_layers
        assigner = LayerDecayValueAssigner(
            list(0.75 ** (num_layers + 1 - i) for i in range(num_layers + 2)))

        if assigner is not None:
            print("Assigned values = %s" % str(assigner.values))
        skip_weight_decay_list = self.model.no_weight_decay()
        print("Skip weight decay list: ", skip_weight_decay_list)

        optimizer = create_optimizer(self.cfg.optimizer, self.model,
                                     skip_list=skip_weight_decay_list,
                                     get_num_layer=assigner.get_layer_id if assigner is not None else None,
                                     get_layer_scale=assigner.get_scale if assigner is not None else None
                                     )

        data_size = len(self.dm.dataset_train)
        num_training_steps_per_epoch = (
            data_size // self.cfg.machine.batch_size // self.cfg.machine.num_devices
        )
        schedule = cosine_scheduler(
            base_value=self.cfg.optimizer.lr,
            final_value=self.cfg.optimizer.min_lr,
            epochs=self.cfg.training.epochs,
            niter_per_ep=num_training_steps_per_epoch,
            warmup_epochs=self.cfg.training.warmup_epochs,
            start_warmup_value=self.cfg.optimizer.min_lr,
            warmup_steps=-1
        )
        # step based lr scheduler
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: schedule[step]/self.cfg.optimizer.lr,
        )
        return [optimizer], [lr_scheduler]


def run(cfg: DictConfig):
    model = RegionLitModel(cfg)
    print(OmegaConf.to_yaml(cfg))
    dm = ReferenceRegionDataModule(cfg)
    model.dm = dm
    wandb = WandbLogger(name=cfg.wandb.run_name,
                        project=cfg.wandb.project_name)
    wandb.log_hyperparams(dict(cfg))
    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="gpu",
        num_sanity_val_steps=0,
        strategy='ddp_spawn',
        devices=cfg.machine.num_devices,
        logger=[wandb,
                CSVLogger('logs', f'{cfg.wandb.project_name}_{cfg.wandb.run_name}')],
        callbacks=[ModelCheckpoint(
            monitor="val_loss", mode="min", save_top_k=1, save_last=True, filename="best")],
        plugins=[MixedPrecision(precision='16-mixed', device="cuda")],
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        gradient_clip_val=cfg.training.clip_grad,
        log_every_n_steps=4,
        val_check_interval=0.5,
        default_root_dir=cfg.machine.output_dir,
    )
    if cfg.stage == 'fit':
        trainer.fit(model, dm)
    if cfg.stage == 'validate':
        trainer.validate(model, datamodule=dm)
    if cfg.stage == 'predict':
        trainer.predict(model, datamodule=dm)
