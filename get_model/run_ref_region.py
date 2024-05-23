import logging

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
from omegaconf import MISSING, DictConfig, OmegaConf
from minlora import add_lora, merge_lora
import wandb
from get_model.config.config import *
from get_model.dataset.zarr_dataset import (
    InferenceReferenceRegionDataset,
    PerturbationInferenceReferenceRegionDataset, ReferenceRegionDataset,
    ReferenceRegionMotif, ReferenceRegionMotifConfig)
from get_model.model.model_refactored import *
from get_model.model.modules import *
from get_model.optim import LayerDecayValueAssigner, create_optimizer
from get_model.run import GETDataModule, LitModel
from get_model.utils import (cosine_scheduler, load_checkpoint, recursive_detach, recursive_numpy, recursive_save_to_zarr, remove_keys,
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

    def build_inference_reference_region_dataset(self, zarr_dataset):
        return InferenceReferenceRegionDataset(self.reference_region_motif, zarr_dataset, quantitative_atac=self.cfg.dataset.quantitative_atac, sampling_step=self.cfg.dataset.sampling_step)

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

    def predict_step(self, batch, batch_idx, *args, **kwargs):
        if self.cfg.task.test_mode == 'inference':
            try:
                loss, pred, obs = self._shared_step(
                    batch, batch_idx, stage='predict')
                goi_idx = batch['tss_peak']
                strand = batch['strand']
                atpm = batch['region_motif'][0][goi_idx, -1].cpu().item()
                gene_name = batch['gene_name'][0]
                for key in pred:
                    pred[key] = pred[key][0][:, strand][goi_idx].max()
                    obs[key] = obs[key][0][:, strand][goi_idx].mean()
                    # save key, pred[key], obs[key] to a csv
                    with open(f"{self.cfg.machine.output_dir}/{self.cfg.wandb.run_name}.csv", "a") as f:
                        f.write(
                            f"{gene_name},{key},{pred[key]},{obs[key]},{atpm}\n")
            except Exception as e:
                print(e)
        elif self.cfg.task.test_mode == 'perturb':
            # try:
            pred = self.perturb_step(batch, batch_idx)
            batch_size = len(batch['WT']['strand'])
            results = []

            for i in range(batch_size):
                strand = batch['WT']['strand'][i].cpu().numpy()
                gene_name = batch['WT']['gene_name'][i]
                tss_peak = batch['WT']['tss_peak'][i].cpu().numpy()
                all_tss_peak = batch['WT']['all_tss_peak'][i].cpu().numpy()
                all_tss_peak = all_tss_peak[all_tss_peak > 0]
                all_tss_peak = all_tss_peak[all_tss_peak < pred['pred_wt']['exp'][i].shape[0]]

                result = {
                    'gene_name': gene_name,
                    'strand': strand,
                    'tss_peak': tss_peak,
                    'perturb_chrom': batch['MUT']['perturb_chrom'][i],
                    'perturb_start': batch['MUT']['perturb_start'][i].cpu().item(),
                    'perturb_end': batch['MUT']['perturb_end'][i].cpu().item(),
                    'pred_wt': pred['pred_wt']['exp'][i][:, strand][all_tss_peak].mean().cpu().item(),
                    'pred_mut': pred['pred_mut']['exp'][i][:, strand][all_tss_peak].mean().cpu().item(),
                    'obs': pred['obs_wt']['exp'][i][:, strand][all_tss_peak].mean().cpu().item()
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
            all_tss_peak = batch['all_tss_peak'][0].cpu().numpy()
            tss_peak = batch['tss_peak'][0].cpu().numpy()
            # assume focus is the center peaks in the input sample
            torch.set_grad_enabled(True)
            pred, obs, jacobians, embeddings = self.interpret_step(batch, batch_idx, layer_names=self.cfg.task.layer_names, focus=tss_peak)
            pred = np.array([pred['exp'][i][:, batch['strand'][i].cpu().numpy()][batch['all_tss_peak'][i].cpu().numpy()].mean() for i in range(len(batch['gene_name']))])
            obs = np.array([obs['exp'][i][:, batch['strand'][i].cpu().numpy()][batch['all_tss_peak'][i].cpu().numpy()].mean() for i in range(len(batch['gene_name']))])
            gene_names = recursive_numpy(recursive_detach(batch['gene_name']))
            for i, gene_name in enumerate(gene_names):
                if len(gene_name)< 20:
                    gene_names[i] = gene_name + ' '*(15-len(gene_name))
            chromosomes = recursive_numpy(recursive_detach(batch['chromosome']))
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
            recursive_save_to_zarr(z, result,  object_codec=object_codec, overwrite=True)
        
        elif self.cfg.task.test_mode == 'interpret_captum':
            tss_peak = batch['tss_peak'][0].cpu().numpy()
            # assume focus is the center peaks in the input sample
            torch.set_grad_enabled(True)
            self.interpret_captum_step(batch, batch_idx, focus=tss_peak)


    def get_model(self):
        model = instantiate(self.cfg.model)
        if self.cfg.finetune.checkpoint is not None:
            checkpoint_model = load_checkpoint(
                self.cfg.finetune.checkpoint)
            state_dict = model.state_dict()
            # remove_keys(checkpoint_model, state_dict)
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
        
        # Add LoRA to the model if specified in the configuration
        if self.cfg.finetune.use_lora:
            add_lora(model)
            
            # Load LoRA parameters based on the stage
            if self.cfg.stage == 'fit':
                # Load LoRA parameters for training
                if self.cfg.finetune.lora_checkpoint is not None:
                    lora_state_dict = torch.load(self.cfg.finetune.lora_checkpoint)
                    model.load_state_dict(rename_lit_state_dict(lora_state_dict['state_dict']), strict=True)
            elif self.cfg.stage in ['validate', 'predict']:
                # Load LoRA parameters for validation and prediction
                if self.cfg.finetune.lora_checkpoint is not None:
                    lora_state_dict = torch.load(self.cfg.finetune.lora_checkpoint)
                    model.load_state_dict(rename_lit_state_dict(lora_state_dict['state_dict']), strict=True)
                    # merge_lora(model)  # Merge LoRA parameters into the model
        
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
        trainer.fit(model, dm)
    if cfg.stage == 'validate':
        trainer.validate(model, datamodule=dm)
    if cfg.stage == 'predict':
        trainer.predict(model, datamodule=dm)


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
