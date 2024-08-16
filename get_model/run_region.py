import logging
from functools import partial
from typing import Callable

import lightning as L
import pandas as pd
import seaborn as sns
import torch
import torch.utils.data
import zarr
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.plugins import MixedPrecision
from matplotlib import pyplot as plt
from minlora import LoRAParametrization
from minlora.model import add_lora_by_name
from omegaconf import MISSING, DictConfig, OmegaConf
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

import wandb
from get_model.config.config import *
from get_model.dataset.zarr_dataset import (InferenceRegionDataset,
                                            RegionDataset, get_gencode_obj)
from get_model.model.model_refactored import *
from get_model.model.modules import *
from get_model.optim import LayerDecayValueAssigner, create_optimizer
from get_model.run import LitModel, get_insulation_overlap
from get_model.utils import (cosine_scheduler, extract_state_dict,
                             load_checkpoint, load_state_dict,
                             recursive_concat_numpy, recursive_detach,
                             recursive_numpy, recursive_save_to_zarr,
                             rename_state_dict)


class RegionDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.accumulated_results = []

    def build_training_dataset(self, is_train=True):
        return RegionDataset(**self.cfg.dataset, is_train=is_train)

    def build_inference_dataset(self, is_train=False, gene_list=None, gencode_obj=None):
        if gencode_obj is None:
            gencode_obj = get_gencode_obj(
                self.cfg.assembly, self.cfg.machine.data_path)

        return InferenceRegionDataset(**self.cfg.dataset, is_train=is_train, gene_list=self.cfg.task.gene_list if gene_list is None else gene_list, gencode_obj=gencode_obj)

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
            elif self.cfg.task.test_mode == 'interpret' or self.cfg.task.test_mode == 'inference':
                self.mutations = None
                self.dataset_predict = self.build_inference_dataset()

        if stage == 'validate':
            self.dataset_val = self.build_training_dataset(is_train=False)

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
        self.accumulated_results = []

    def optimizer_step(self, epoch: int, batch_idx: int, optimizer: Optimizer | LightningOptimizer, optimizer_closure: Callable[[], Any] | None = None) -> None:
        self.lr_schedulers().step()
        return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    def validation_step(self, batch, batch_idx):
        loss, pred, obs = self._shared_step(batch, batch_idx, stage='val')
        # print(pred['exp'].detach().cpu().numpy().flatten().max(),
        #       obs['exp'].detach().cpu().numpy().flatten().max())

        if self.cfg.eval_tss and 'exp' in pred:
            tss_idx = batch['mask']
            for key in pred:
                # pred[key] = (pred[key] * tss_idx)
                # obs[key] = (obs[key] * tss_idx)
                pred[key] = pred[key][tss_idx > 0].flatten()
                obs[key] = obs[key][tss_idx > 0].flatten()
            # error handling in metric when there is no TSS, or all TSS is not expressed in batch.
            if pred['exp'].shape[0] == 0:
                return
            if obs['exp'].sum() == 0:
                return
            # check for nan
            if torch.isnan(pred['exp']).any() or torch.isnan(obs['exp']).any():
                return

        metrics = self.metrics(pred, obs)
        if batch_idx == 0 and self.cfg.log_image:
            # log one example as scatter plot
            for key in pred:
                plt.clf()
                self.logger.experiment.log({
                    f"scatter_{key}": wandb.Image(sns.scatterplot(y=pred[key].detach().cpu().numpy().flatten(), x=obs[key].detach().cpu().numpy().flatten()))
                })
        distributed = self.cfg.machine.num_devices > 1
        self.log_dict(
            metrics, batch_size=self.cfg.machine.batch_size, sync_dist=distributed)
        self.log("val_loss", loss,
                 batch_size=self.cfg.machine.batch_size, sync_dist=distributed)

    def predict_step(self, batch, batch_idx, *args, **kwargs):
        if self.cfg.task.test_mode == 'inference':
            loss, preds, obs = self._shared_step(
                batch, batch_idx, stage='predict')
            result_df = []
            for batch_element in range(len(batch['gene_name'])):
                goi_idx = batch['all_tss_peak'][batch_element]
                goi_idx = goi_idx[goi_idx > 0]  # filter out pad (tss_peak = 0)
                strand = batch['strand'][batch_element]
                atpm = batch['region_motif'][batch_element][goi_idx, -
                                                            1].max().cpu().item()
                gene_name = batch['gene_name'][batch_element]
                for key in preds:
                    result_df.append(
                        {'gene_name': gene_name, 'key': key, 'pred': preds[key][batch_element][:, strand][goi_idx].max().cpu().item(), 'obs': obs[key][batch_element][:, strand][goi_idx].max().cpu().item(), 'atpm': atpm})
            result_df = pd.DataFrame(result_df)
            # mkdir if not exist
            os.makedirs(
                f"{self.cfg.machine.output_dir}/{self.cfg.wandb.project_name}", exist_ok=True)
            result_df.to_csv(
                f"{self.cfg.machine.output_dir}/{self.cfg.wandb.project_name}/{self.cfg.wandb.run_name}.csv", index=False, mode='a', header=False
            )
        elif self.cfg.task.test_mode == 'perturb':
            # TODO: need to figure out if batching is working
            preds = self.perturb_step(batch, batch_idx)
            batch_size = len(batch['WT']['strand'])
            results = []

            for i in range(batch_size):
                strand = batch['WT']['strand'][i].cpu().numpy()
                gene_name = batch['WT']['gene_name'][i]
                tss_peak = batch['WT']['tss_peak'][i].cpu().numpy()
                goi_idx = batch['WT']['all_tss_peak'][i].cpu().numpy()
                goi_idx = goi_idx[goi_idx > 0]
                goi_idx = goi_idx[goi_idx <
                                  preds['pred_wt']['exp'][i].shape[0]]

                result = {
                    'gene_name': gene_name,
                    'strand': strand,
                    'tss_peak': tss_peak,
                    'perturb_chrom': batch['MUT']['perturb_chrom'][i],
                    'perturb_start': batch['MUT']['perturb_start'][i].cpu().item(),
                    'perturb_end': batch['MUT']['perturb_end'][i].cpu().item(),
                    'pred_wt': preds['pred_wt']['exp'][i][:, strand][goi_idx].mean().cpu().item(),
                    'pred_mut': preds['pred_mut']['exp'][i][:, strand][goi_idx].mean().cpu().item(),
                    'obs': preds['obs_wt']['exp'][i][:, strand][goi_idx].mean().cpu().item()
                }
                results.append(result)

            # Save results to a csv as multiple rows
            results_df = pd.DataFrame(results)
            results_df.to_csv(
                f"{self.cfg.machine.output_dir}/{self.cfg.wandb.run_name}.csv", index=False, mode='a', header=False
            )
            # except Exception as    e:
            # print(e)

        elif self.cfg.task.test_mode == 'interpret':
            focus = []
            for i in range(len(batch['gene_name'])):
                goi_idx = batch['all_tss_peak'][i].cpu().numpy()
                goi_idx = goi_idx[goi_idx > 0]
                focus.append(goi_idx)
            torch.set_grad_enabled(True)
            preds, obs, jacobians, embeddings = self.interpret_step(
                batch, batch_idx, layer_names=self.cfg.task.layer_names, focus=focus)
            # pred = np.array([pred['exp'][i][:, batch['strand'][i].cpu().numpy(
            # )][batch['all_tss_peak'][i].cpu().numpy()].mean() for i in range(len(batch['gene_name']))])
            # obs = np.array([obs['exp'][i][:, batch['strand'][i].cpu().numpy(
            # )][batch['all_tss_peak'][i].cpu().numpy()].mean() for i in range(len(batch['gene_name']))])
            gene_names = recursive_numpy(recursive_detach(batch['gene_name']))
            for i, gene_name in enumerate(gene_names):
                if len(gene_name) < 100:
                    gene_names[i] = gene_name + ' '*(100-len(gene_name))
            chromosomes = recursive_numpy(
                recursive_detach(batch['chromosome']))
            for i, chromosome in enumerate(chromosomes):
                if len(chromosome) < 30:
                    chromosomes[i] = chromosome + ' '*(30-len(chromosome))

            result = {
                'preds': preds,
                'obs': obs,
                'jacobians': jacobians,
                'input': embeddings['input']['region_motif'],
                'chromosome': chromosomes,
                'peak_coord': recursive_numpy(recursive_detach(batch['peak_coord'])),
                'strand': recursive_numpy(recursive_detach(batch['strand'])),
                'focus': recursive_numpy(recursive_detach(batch['all_tss_peak'])),
                'avaliable_genes': gene_names,
            }
            self.accumulated_results.append(result)

        elif self.cfg.task.test_mode == 'interpret_captum':
            tss_peak = batch['tss_peak'][0].cpu().numpy()

            new_peak_start_idx, new_peak_end_idx, new_tss_peak = get_insulation_overlap(
                batch, self.dm.dataset_predict.zarr_dataset.datapool.insulation)
            for shift in np.random.randint(-10, 10, 5):
                if new_peak_start_idx+shift < 0 or new_peak_end_idx+shift >= batch['region_motif'][0].shape[0]:
                    continue
                # assume focus is the center peaks in the input sample
                torch.set_grad_enabled(True)
                self.interpret_captum_step(
                    batch, batch_idx, focus=new_tss_peak, start=new_peak_start_idx, end=new_peak_end_idx, shift=shift)

    def get_model(self):
        model = instantiate(self.cfg.model)

        # Load main model checkpoint
        if self.cfg.finetune.checkpoint is not None:
            checkpoint_model = load_checkpoint(
                self.cfg.finetune.checkpoint, model_key=self.cfg.finetune.model_key)
            checkpoint_model = extract_state_dict(checkpoint_model)
            checkpoint_model = rename_state_dict(
                checkpoint_model, self.cfg.finetune.rename_config)
            lora_config = {  # specify which layers to add lora to, by default only add to linear layers
                nn.Linear: {
                    "weight": partial(LoRAParametrization.from_linear, rank=8),
                },
                nn.Conv2d: {
                    "weight": partial(LoRAParametrization.from_conv2d, rank=4),
                },
            }
            if any("lora" in k for k in checkpoint_model.keys()) and self.cfg.finetune.use_lora:
                add_lora_by_name(
                    model, self.cfg.finetune.layers_with_lora, lora_config)
                load_state_dict(model, checkpoint_model,
                                strict=self.cfg.finetune.strict)
            elif any("lora" in k for k in checkpoint_model.keys()) and not self.cfg.finetune.use_lora:
                raise ValueError(
                    "Model checkpoint contains LoRA parameters but use_lora is set to False")
            elif not any("lora" in k for k in checkpoint_model.keys()) and self.cfg.finetune.use_lora:
                logging.info(
                    "Model checkpoint does not contain LoRA parameters but use_lora is set to True, using the checkpoint as base model")
                load_state_dict(model, checkpoint_model,
                                strict=self.cfg.finetune.strict)
                add_lora_by_name(
                    model, self.cfg.finetune.layers_with_lora, lora_config)
            else:
                load_state_dict(model, checkpoint_model,
                                strict=self.cfg.finetune.strict)

        # Load additional checkpoints
        if len(self.cfg.finetune.additional_checkpoints) > 0:
            for checkpoint_config in self.cfg.finetune.additional_checkpoints:
                checkpoint_model = load_checkpoint(
                    checkpoint_config.checkpoint, model_key=checkpoint_config.model_key)
                checkpoint_model = extract_state_dict(checkpoint_model)
                checkpoint_model = rename_state_dict(
                    checkpoint_model, checkpoint_config.rename_config)
                load_state_dict(model, checkpoint_model,
                                strict=checkpoint_config.strict)

        if self.cfg.finetune.use_lora:
            # Load LoRA parameters based on the stage
            if self.cfg.stage == 'fit':
                # Load LoRA parameters for training
                if self.cfg.finetune.lora_checkpoint is not None:
                    lora_state_dict = load_checkpoint(
                        self.cfg.finetune.lora_checkpoint)
                    lora_state_dict = extract_state_dict(lora_state_dict)
                    lora_state_dict = rename_state_dict(
                        lora_state_dict, self.cfg.finetune.lora_rename_config)
                    load_state_dict(model, lora_state_dict, strict=True)
            elif self.cfg.stage in ['validate', 'predict']:
                # Load LoRA parameters for validation and prediction
                if self.cfg.finetune.lora_checkpoint is not None:
                    lora_state_dict = load_checkpoint(
                        self.cfg.finetune.lora_checkpoint)
                    lora_state_dict = extract_state_dict(lora_state_dict)
                    lora_state_dict = rename_state_dict(
                        lora_state_dict, self.cfg.finetune.lora_rename_config)
                    load_state_dict(model, lora_state_dict, strict=True)

        model.freeze_layers(
            patterns_to_freeze=self.cfg.finetune.patterns_to_freeze, invert_match=False)
        print("Model = %s" % str(model))
        return model

    def on_validation_epoch_end(self):
        pass

    def on_predict_epoch_end(self):
        if self.cfg.task.test_mode == 'interpret':
            # Save accumulated results to zarr
            zarr_path = f"{self.cfg.machine.output_dir}/{self.cfg.wandb.project_name}/{self.cfg.wandb.run_name}.zarr"
            from numcodecs import VLenUTF8
            object_codec = VLenUTF8()
            z = zarr.open(zarr_path, mode='w')

            accumulated_results = recursive_concat_numpy(
                self.accumulated_results)
            recursive_save_to_zarr(
                z, accumulated_results, object_codec=object_codec, overwrite=True)

            # Clear accumulated results
            self.accumulated_results = []

    def configure_optimizers(self):
        if hasattr(self.model.cfg, 'encoder'):
            num_layers = self.model.cfg.encoder.num_layers
        else:
            num_layers = 0
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
    dm = RegionDataModule(cfg)
    model.dm = dm
    wandb_logger = WandbLogger(name=cfg.wandb.run_name,
                               project=cfg.wandb.project_name)
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    if cfg.machine.num_devices > 0:
        strategy = 'auto'
        accelerator = 'gpu'
        device = cfg.machine.num_devices
        if cfg.machine.num_devices > 1:
            strategy = 'ddp_find_unused_parameters_true'
    else:
        strategy = 'auto'
        accelerator = 'cpu'
        device = 'auto'
    inference_mode = True
    if 'interpret' in cfg.task.test_mode:
        inference_mode = False
    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=accelerator,
        num_sanity_val_steps=10,
        strategy=strategy,
        devices=device,
        logger=[
            wandb_logger,
            CSVLogger('logs', f'{cfg.wandb.project_name}_{cfg.wandb.run_name}')],
        callbacks=[ModelCheckpoint(
            monitor="val_loss", mode="min", save_top_k=1, save_last=True, filename="best")],
        # plugins=[MixedPrecision(precision='16-mixed', device="cuda")],
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        gradient_clip_val=cfg.training.clip_grad,
        default_root_dir=cfg.machine.output_dir,
        inference_mode=inference_mode
    )
    if cfg.stage == 'fit':
        trainer.fit(model, dm)
    if cfg.stage == 'validate':
        trainer.validate(model, datamodule=dm)
    if cfg.stage == 'predict':
        trainer.predict(model, datamodule=dm)
