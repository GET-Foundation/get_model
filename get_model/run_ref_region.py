import logging
from functools import partial

import lightning as L
import pandas as pd
import seaborn as sns
import torch
import torch.utils.data
import zarr
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.plugins import MixedPrecision
from matplotlib import pyplot as plt
from minlora import LoRAParametrization
from minlora.model import add_lora_by_name
from omegaconf import MISSING, DictConfig, OmegaConf

import wandb
from get_model.config.config import *
from get_model.dataset.collate import everything_collate
from get_model.dataset.zarr_dataset import (
    EverythingDataset,
    InferenceEverythingDataset,
    InferenceReferenceRegionDataset,
    PerturbationInferenceReferenceRegionDataset, ReferenceRegionDataset,
    ReferenceRegionMotif, ReferenceRegionMotifConfig)
from get_model.model.model_refactored import *
from get_model.model.modules import *
from get_model.optim import LayerDecayValueAssigner, create_optimizer
from get_model.run import GETDataModule, LitModel, get_insulation_overlap
from get_model.utils import (cosine_scheduler, extract_state_dict,
                             load_checkpoint, load_state_dict,
                             recursive_detach, recursive_numpy,
                             recursive_save_to_zarr, rename_state_dict)


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

    def build_inference_reference_region_dataset(self, zarr_dataset):
        return InferenceReferenceRegionDataset(self.reference_region_motif, zarr_dataset, quantitative_atac=self.cfg.dataset.quantitative_atac, sampling_step=self.cfg.dataset.sampling_step)

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
            shuffle=False,
        )


class EverythingDataModule(ReferenceRegionDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        logging.info("Init EverythingDataModule")

    def build_from_zarr_dataset(self, zarr_dataset):
        return EverythingDataset(self.reference_region_motif, zarr_dataset, quantitative_atac=self.cfg.dataset.quantitative_atac, sampling_step=self.cfg.dataset.sampling_step)

    def build_inference_reference_region_dataset(self, zarr_dataset):
        return InferenceEverythingDataset(self.reference_region_motif, zarr_dataset, quantitative_atac=self.cfg.dataset.quantitative_atac, sampling_step=self.cfg.dataset.sampling_step)

    def build_perturbation_inference_dataset(self, zarr_dataset, perturbations, mode='peak_inactivation'):
        inference_dataset = self.build_inference_reference_region_dataset(
            zarr_dataset)
        print("Perturbations mode", mode)
        return PerturbationInferenceReferenceRegionDataset(inference_dataset, perturbations, mode=mode)

    def setup(self, stage=None):
        super().setup(stage)
        if stage == 'fit' or stage is None:
            self.dataset_train = self.build_from_zarr_dataset(
                self.dataset_train)
            self.dataset_val = self.build_from_zarr_dataset(self.dataset_val)
        if stage == 'predict':
            if self.cfg.task.test_mode == 'predict':
                self.dataset_predict = self.build_from_zarr_dataset(
                    self.dataset_predict)
            elif self.cfg.task.test_mode == 'inference' or self.cfg.task.test_mode == 'interpret' or self.cfg.task.test_mode == 'interpret_captum':
                self.dataset_predict = self.build_inference_reference_region_dataset(
                    self.dataset_predict)
            elif 'perturb' in self.cfg.task.test_mode:
                self.mutations = pd.read_csv(self.cfg.task.mutations, sep='\t')
                perturb_mode = 'mutation' if 'mutation' in self.cfg.task.test_mode else 'peak_inactivation'
                self.dataset_predict = self.build_perturbation_inference_dataset(
                    self.dataset_predict.inference_dataset, self.mutations, mode=perturb_mode
                )
        if stage == 'validate':
            self.dataset_val = self.build_from_zarr_dataset(self.dataset_val)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.cfg.machine.batch_size,
            num_workers=self.cfg.machine.num_workers,
            drop_last=True,
            shuffle=True,
            collate_fn=everything_collate
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.cfg.machine.batch_size,
            num_workers=self.cfg.machine.num_workers,
            drop_last=True,
            collate_fn=everything_collate
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=self.cfg.machine.batch_size,
            num_workers=self.cfg.machine.num_workers,
            drop_last=True,
            collate_fn=everything_collate
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_predict,
            batch_size=self.cfg.machine.batch_size,
            num_workers=self.cfg.machine.num_workers,
            drop_last=False,
            shuffle=False,
            collate_fn=everything_collate
        )


class RegionLitModel(LitModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.min_exp_loss = float('inf')
        self.exp_overfit_count = 0
        self.exp_overfit_threshold = 1

    def validation_step(self, batch, batch_idx):
        loss, pred, obs = self._shared_step(batch, batch_idx, stage='val')
        # print(pred['exp'].detach().cpu().numpy().flatten().max(),
        #       obs['exp'].detach().cpu().numpy().flatten().max())

        # if size of obs is too large, subsample 2000 elements
        if 'hic' in obs:
            idx = np.random.choice(
                obs['hic'].flatten().shape[0], 1000, replace=False)
            obs['hic'] = obs['hic'].flatten()[idx]
            pred['hic'] = pred['hic'].flatten()[idx]

        if 'abc' in obs:
            idx = np.random.choice(
                obs['abc'].flatten().shape[0], 1000, replace=False)
            obs['abc'] = obs['abc'].flatten()[idx]
            pred['abc'] = pred['abc'].flatten()[idx]

        if 'motif' in obs:
            # random draw 1 sample, discard the rest
            idx = np.random.choice(
                obs['motif'].shape[0], 1, replace=False)
            obs['motif'] = obs['motif'][idx]
            pred['motif'] = pred['motif'][idx]

        if 'exp' in obs and self.cfg.eval_tss:
            tss_idx = batch['mask']
            # pred[key] = (pred[key] * tss_idx)
            # obs[key] = (obs[key] * tss_idx)
            pred['exp'] = pred['exp'][tss_idx > 0].flatten()
            obs['exp'] = obs['exp'][tss_idx > 0].flatten()
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
        # if distributed, set sync_dist=True
        distributed = self.cfg.machine.num_devices > 1
        self.log_dict(
            metrics, batch_size=self.cfg.machine.batch_size, sync_dist=distributed)
        self.log("val_loss", loss,
                 batch_size=self.cfg.machine.batch_size, sync_dist=distributed)

    def on_train_epoch_end(self):
        # get loss from the last epoch
        if 'val_exp_loss' not in self.trainer.callback_metrics:
            return
        val_exp_loss = self.trainer.callback_metrics['val_exp_loss']
        if val_exp_loss < self.min_exp_loss:
            self.min_exp_loss = val_exp_loss
        else:
            self.exp_overfit_count += 1
        if self.exp_overfit_count > self.exp_overfit_threshold:
            # freeze encoder, region_embed, head_exp
            # self.model.freeze_layers(
            #     patterns_to_freeze=['encoder', 'region_embed', 'head_exp'], invert_match=False)
            print("Freezing exp loss component")
            self.loss.freeze_component('exp')

    def predict_step(self, batch, batch_idx, *args, **kwargs):
        if self.cfg.task.test_mode == 'inference':
            loss, pred, obs = self._shared_step(
                batch, batch_idx, stage='predict')
            result_df = []
            for batch_element in range(len(batch['gene_name'])):
                goi_idx = batch['all_tss_peak'][batch_element]
                goi_idx = goi_idx[goi_idx > 0]  # filter out pad (tss_peak = 0)
                strand = batch['strand'][batch_element]
                atpm = batch['region_motif'][batch_element][goi_idx, -
                                                            1].max().cpu().item()
                gene_name = batch['gene_name'][batch_element]
                for key in pred:
                    result_df.append(
                        {'gene_name': gene_name, 'key': key, 'pred': pred[key][batch_element][:, strand][goi_idx].max().cpu().item(), 'obs': obs[key][batch_element][:, strand][goi_idx].max().cpu().item(), 'atpm': atpm})
            result_df = pd.DataFrame(result_df)
            result_df.to_csv(
                f"{self.cfg.machine.output_dir}/{self.cfg.wandb.run_name}.csv", index=False, mode='a', header=False
            )
        elif self.cfg.task.test_mode == 'perturb':
            # TODO: need to figure out if batching is working
            pred = self.perturb_step(batch, batch_idx)
            batch_size = len(batch['WT']['strand'])
            results = []

            for i in range(batch_size):
                strand = batch['WT']['strand'][i].cpu().numpy()
                gene_name = batch['WT']['gene_name'][i]
                tss_peak = batch['WT']['tss_peak'][i].cpu().numpy()
                goi_idx = batch['WT']['all_tss_peak'][i].cpu().numpy()
                goi_idx = goi_idx[goi_idx > 0]
                goi_idx = goi_idx[goi_idx < pred['pred_wt']['exp'][i].shape[0]]

                result = {
                    'gene_name': gene_name,
                    'strand': strand,
                    'tss_peak': tss_peak,
                    'perturb_chrom': batch['MUT']['perturb_chrom'][i],
                    'perturb_start': batch['MUT']['perturb_start'][i].cpu().item(),
                    'perturb_end': batch['MUT']['perturb_end'][i].cpu().item(),
                    'pred_wt': pred['pred_wt']['exp'][i][:, strand][goi_idx].mean().cpu().item(),
                    'pred_mut': pred['pred_mut']['exp'][i][:, strand][goi_idx].mean().cpu().item(),
                    'obs': pred['obs_wt']['exp'][i][:, strand][goi_idx].mean().cpu().item()
                }
                results.append(result)

            # Save results to a csv as multiple rows
            results_df = pd.DataFrame(results)
            results_df.to_csv(
                f"{self.cfg.machine.output_dir}/{self.cfg.wandb.run_name}.csv", index=False, mode='a', header=False
            )
            # except Exception as e:
            # print(e)
        elif self.cfg.task.test_mode == 'interpret':
            goi_idx = batch['all_tss_peak'][0].cpu().numpy()
            tss_peak = batch['tss_peak'][0].cpu().numpy()
            # assume focus is the center peaks in the input sample
            torch.set_grad_enabled(True)
            pred, obs, jacobians, embeddings = self.interpret_step(
                batch, batch_idx, layer_names=self.cfg.task.layer_names, focus=tss_peak)
            pred = np.array([pred['exp'][i][:, batch['strand'][i].cpu().numpy(
            )][batch['all_tss_peak'][i].cpu().numpy()].mean() for i in range(len(batch['gene_name']))])
            obs = np.array([obs['exp'][i][:, batch['strand'][i].cpu().numpy(
            )][batch['all_tss_peak'][i].cpu().numpy()].mean() for i in range(len(batch['gene_name']))])
            gene_names = recursive_numpy(recursive_detach(batch['gene_name']))
            for i, gene_name in enumerate(gene_names):
                if len(gene_name) < 20:
                    gene_names[i] = gene_name + ' '*(15-len(gene_name))
            chromosomes = recursive_numpy(
                recursive_detach(batch['chromosome']))
            for i, chromosome in enumerate(chromosomes):
                if len(chromosome) < 10:
                    chromosomes[i] = chromosome + ' '*(10-len(chromosome))

            result = {
                'pred': pred,
                'obs': obs,
                'jacobians': jacobians,
                'input': embeddings['input']['region_motif'],
                'peaks': recursive_numpy(recursive_detach(batch['peak_coord'])),
                'chromosome': chromosomes,
                'strand': recursive_numpy(recursive_detach(batch['strand'])),
                'gene_name': gene_names,
            }
            # save to zarr
            zarr_path = f"{self.cfg.machine.output_dir}/{self.cfg.wandb.run_name}.zarr"
            from numcodecs import VLenUTF8
            object_codec = VLenUTF8()
            z = zarr.open(zarr_path, mode='a')
            recursive_save_to_zarr(
                z, result,  object_codec=object_codec, overwrite=True)

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
        # save self.trainer.callback_metrics to a csv as one row
        metric_dict = dict_to_item(self.trainer.callback_metrics)
        metric_dict['peak_count_filter'] = self.cfg.dataset.peak_count_filter
        metric_dict['motif_scaler'] = self.cfg.dataset.reference_region_motif.motif_scaler
        metric_dict['quantitative_atac'] = self.cfg.dataset.quantitative_atac
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
        if hasattr(self.model.cfg, 'encoder') and hasattr(self.model.cfg.encoder, 'num_layers'):
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


def _run(cfg: DictConfig, dm: ReferenceRegionDataModule):
    model = RegionLitModel(cfg)
    print(OmegaConf.to_yaml(cfg))
    model.dm = dm
    wandb_logger = WandbLogger(name=cfg.wandb.run_name,
                               project=cfg.wandb.project_name)
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    if cfg.machine.num_devices > 0:
        strategy = 'auto'
        accelerator = 'gpu'
        device = cfg.machine.num_devices
        if cfg.machine.num_devices > 1:
            strategy = 'ddp_spawn'
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
        num_sanity_val_steps=0,
        strategy=strategy,
        devices=device,
        logger=[
            wandb_logger,
            CSVLogger('logs', f'{cfg.wandb.project_name}_{cfg.wandb.run_name}')],
        callbacks=[ModelCheckpoint(
            monitor="val_loss", mode="min", save_top_k=1, save_last=True, filename="best")],
        plugins=[MixedPrecision(precision='16-mixed', device="cuda")],
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        gradient_clip_val=cfg.training.clip_grad,
        log_every_n_steps=4,
        val_check_interval=0.5,
        default_root_dir=cfg.machine.output_dir,
        inference_mode=inference_mode
    )
    if cfg.stage == 'fit':
        trainer.fit(model, datamodule=dm, ckpt_path=cfg.finetune.resume_ckpt)
    if cfg.stage == 'validate':
        trainer.validate(model, datamodule=dm,
                         ckpt_path=cfg.fintune.resume_ckpt)
    if cfg.stage == 'predict':
        trainer.predict(model, datamodule=dm,
                        ckpt_path=cfg.finetune.resume_ckpt)


def run_everything(cfg: DictConfig):
    dm = EverythingDataModule(cfg)
    _run(cfg, dm)


def run_ref_region(cfg: DictConfig):
    dm = ReferenceRegionDataModule(cfg)
    _run(cfg, dm)


# def run_downstream(cfg: DictConfig):
#     torch.set_float32_matmul_precision('medium')
#     model = LitModel(cfg)
#     # move the model to the gpu
#     model.to('cuda')
#     dm = GETDataModule(cfg)
#     model.dm = dm
#     if cfg.machine.num_devices > 0:
#         strategy = 'auto'
#         accelerator = 'gpu'
#         device = cfg.machine.num_devices
#         if cfg.machine.num_devices > 1:
#             strategy = 'ddp_spawn'
#     else:
#         strategy = 'auto'
#         accelerator = 'cpu'
#         device = 'auto'
#     trainer = L.Trainer(
#         max_epochs=cfg.training.epochs,
#         accelerator=accelerator,
#         num_sanity_val_steps=10,
#         strategy=strategy,
#         devices=device,
#         # plugins=[MixedPrecision(precision='16-mixed', device="cuda")],
#         accumulate_grad_batches=cfg.training.accumulate_grad_batches,
#         gradient_clip_val=cfg.training.clip_grad,
#         log_every_n_steps=100,
#         deterministic=True,
#         default_root_dir=cfg.machine.output_dir,
#     )
#     print(run_ppif_task(trainer, model))
