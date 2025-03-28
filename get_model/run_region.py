import gc
import logging
from functools import partial

import lightning as L
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.utils.data
import wandb
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from minlora import LoRAParametrization
from minlora.model import add_lora_by_name
from omegaconf import DictConfig, OmegaConf

from get_model.config.config import *
from get_model.dataset.zarr_dataset import (
    InferenceRegionDataset,
    InferenceRegionMotifDataset,
    RegionDataset,
    RegionMotifDataset,
    get_gencode_obj,
)
from get_model.model.model import *
from get_model.model.modules import *
from get_model.run import LitModel, run_shared
from get_model.utils import (
    extract_state_dict,
    load_checkpoint,
    load_state_dict,
    recursive_detach,
    recursive_numpy,
    rename_state_dict,
)


class RegionDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.accumulated_results = []

    def build_training_dataset(self, is_train=True):
        return RegionDataset(**self.cfg.dataset, is_train=is_train)

    def build_inference_dataset(self, is_train=False, gene_list=None, gencode_obj=None):
        if gencode_obj is None:
            gencode_obj = get_gencode_obj(self.cfg.assembly, self.cfg.machine.data_path)

        return InferenceRegionDataset(
            **self.cfg.dataset,
            is_train=is_train,
            gene_list=self.cfg.task.gene_list if gene_list is None else gene_list,
            gencode_obj=gencode_obj,
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.dataset_train = self.build_training_dataset(is_train=True)
            self.dataset_val = self.build_training_dataset(is_train=False)
        if stage == "predict":
            if self.cfg.task.test_mode == "predict":
                self.dataset_predict = self.build_training_dataset(is_train=False)
            elif (
                self.cfg.task.test_mode == "interpret"
                or self.cfg.task.test_mode == "inference"
            ):
                self.mutations = None
                self.dataset_predict = self.build_inference_dataset()

        if stage == "validate":
            self.dataset_val = self.build_training_dataset(is_train=False)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.cfg.machine.batch_size,
            num_workers=self.cfg.machine.num_workers,
            drop_last=True,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.cfg.machine.batch_size,
            num_workers=self.cfg.machine.num_workers,
            drop_last=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=self.cfg.machine.batch_size,
            num_workers=self.cfg.machine.num_workers,
            drop_last=True,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_predict,
            batch_size=self.cfg.machine.batch_size,
            num_workers=self.cfg.machine.num_workers,
            drop_last=False,
            persistent_workers=True,
        )


class RegionLitModel(LitModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def validation_step(self, batch, batch_idx):
        loss, pred, obs = self._shared_step(batch, batch_idx, stage="val")
        # logging.debug(pred['exp'].detach().cpu().numpy().flatten().max(),
        #       obs['exp'].detach().cpu().numpy().flatten().max())

        if self.cfg.eval_tss and "exp" in pred:
            tss_idx = batch["mask"]
            for key in pred:
                # pred[key] = (pred[key] * tss_idx)
                # obs[key] = (obs[key] * tss_idx)
                pred[key] = pred[key][tss_idx > 0].flatten()
                obs[key] = obs[key][tss_idx > 0].flatten()
            # error handling in metric when there is no TSS, or all TSS is not expressed in batch.
            if pred["exp"].shape[0] == 0:
                return
            if obs["exp"].sum() == 0:
                return
            # check for nan
            if torch.isnan(pred["exp"]).any() or torch.isnan(obs["exp"]).any():
                return

        if "hic" in obs:
            if obs["hic"].flatten().shape[0] > 1000:
                try:
                    idx = np.random.choice(
                        obs["hic"].flatten().shape[0], 1000, replace=False
                    )
                    obs["hic"] = obs["hic"].flatten()[idx]
                    pred["hic"] = pred["hic"].flatten()[idx]
                except Exception as e:
                    logging.debug(obs["hic"].shape)
                    logging.debug(pred["hic"].shape)
            else:
                obs["hic"] = torch.randn(1000).to(obs["hic"].device)
                pred["hic"] = torch.randn(1000).to(pred["hic"].device)
        metrics = self.metrics(pred, obs)
        if batch_idx == 0 and self.cfg.log_image:
            # log one example as scatter plot
            for key in pred:
                plt.clf()
                if self.cfg.run.use_wandb:
                    self.logger.experiment.log(
                        {
                            f"scatter_{key}": wandb.Image(
                                sns.scatterplot(
                                    y=pred[key].detach().cpu().numpy().flatten(),
                                    x=obs[key].detach().cpu().numpy().flatten(),
                                )
                            )
                        }
                    )
        distributed = self.cfg.machine.num_devices > 1
        self.log_dict(
            metrics, batch_size=self.cfg.machine.batch_size, sync_dist=distributed
        )
        self.log(
            "val_loss",
            loss,
            batch_size=self.cfg.machine.batch_size,
            sync_dist=distributed,
        )

    def predict_step(self, batch, batch_idx, *args, **kwargs):
        if self.cfg.task.test_mode == "inference":
            loss, preds, obs = self._shared_step(batch, batch_idx, stage="predict")
            result_df = []
            for batch_element in range(len(batch["gene_name"])):
                goi_idx = batch["all_tss_peak"][batch_element]
                goi_idx = goi_idx[goi_idx > 0]  # filter out pad (tss_peak = 0)
                strand = batch["strand"][batch_element]
                atpm = (
                    batch["region_motif"][batch_element][goi_idx, -1].max().cpu().item()
                )
                gene_name = batch["gene_name"][batch_element]
                for key in preds:
                    result_df.append(
                        {
                            "gene_name": gene_name,
                            "key": key,
                            "pred": preds[key][batch_element][:, strand][goi_idx]
                            .max()
                            .cpu()
                            .item(),
                            "obs": obs[key][batch_element][:, strand][goi_idx]
                            .max()
                            .cpu()
                            .item(),
                            "atpm": atpm,
                        }
                    )
            result_df = pd.DataFrame(result_df)
            # mkdir if not exist
            os.makedirs(
                f"{self.cfg.machine.output_dir}/{self.cfg.run.project_name}",
                exist_ok=True,
            )
            result_df.to_csv(
                f"{self.cfg.machine.output_dir}/{self.cfg.run.project_name}/{self.cfg.run.run_name}.csv",
                index=False,
                mode="a",
                header=False,
            )
            return result_df
        elif self.cfg.task.test_mode == "perturb":
            # TODO: need to figure out if batching is working
            preds = self.perturb_step(batch, batch_idx)
            batch_size = len(batch["WT"]["strand"])
            results = []

            for i in range(batch_size):
                strand = batch["WT"]["strand"][i].cpu().numpy()
                gene_name = batch["WT"]["gene_name"][i]
                tss_peak = batch["WT"]["tss_peak"][i].cpu().numpy()
                goi_idx = batch["WT"]["all_tss_peak"][i].cpu().numpy()
                goi_idx = goi_idx[goi_idx > 0]
                goi_idx = goi_idx[goi_idx < preds["pred_wt"]["exp"][i].shape[0]]

                result = {
                    "gene_name": gene_name,
                    "strand": strand,
                    "tss_peak": tss_peak,
                    "perturb_chrom": batch["MUT"]["perturb_chrom"][i],
                    "perturb_start": batch["MUT"]["perturb_start"][i].cpu().item(),
                    "perturb_end": batch["MUT"]["perturb_end"][i].cpu().item(),
                    "pred_wt": preds["pred_wt"]["exp"][i][:, strand][goi_idx]
                    .mean()
                    .cpu()
                    .item(),
                    "pred_mut": preds["pred_mut"]["exp"][i][:, strand][goi_idx]
                    .mean()
                    .cpu()
                    .item(),
                    "obs": preds["obs_wt"]["exp"][i][:, strand][goi_idx]
                    .mean()
                    .cpu()
                    .item(),
                }
                results.append(result)

            # Save results to a csv as multiple rows
            results_df = pd.DataFrame(results)
            results_df.to_csv(
                f"{self.cfg.machine.output_dir}/{self.cfg.run.run_name}.csv",
                index=False,
                mode="a",
                header=False,
            )
            return results_df
            # except Exception as    e:
            # logging.debug(e)

        elif self.cfg.task.test_mode == "interpret":
            with torch.enable_grad():
                focus = 100  # Fixed focus point
                preds, obs, jacobians, embeddings = self.interpret_step(
                    batch, batch_idx, layer_names=self.cfg.task.layer_names, focus=focus
                )
                
                # Create padded arrays and reshape to match other dimensions
                gene_names = np.array([
                    name.ljust(100) if len(name) < 100 else name[:100] 
                    for name in batch["gene_name"]
                ], dtype='U100')
                gene_names = gene_names.reshape(-1)  # Flatten from (1000, 8) to (8000,)
                
                chromosomes = np.array([
                    chrom.ljust(30) if len(chrom) < 30 else chrom[:30]
                    for chrom in batch["chromosome"]
                ], dtype='U30')
                chromosomes = chromosomes.reshape(-1)  # Flatten from (1000, 8) to (8000,)

                result = {
                    "preds": preds,
                    "obs": obs,
                    "jacobians": jacobians,
                    "input": embeddings["input"]["region_motif"],
                    "embeddings": embeddings,
                    "chromosome": chromosomes,
                    "peak_coord": recursive_numpy(recursive_detach(batch["peak_coord"])),
                    "strand": recursive_numpy(recursive_detach(batch["strand"])),
                    "focus": recursive_numpy(recursive_detach(batch["all_tss_peak"])),
                    "available_genes": gene_names,
                }
                self.accumulated_results.append(result)
                return

    def get_model(self):
        model = instantiate(self.cfg.model)

        # Load main model checkpoint
        if self.cfg.finetune.checkpoint is not None:
            checkpoint_model = load_checkpoint(
                self.cfg.finetune.checkpoint, model_key=self.cfg.finetune.model_key
            )
            checkpoint_model = extract_state_dict(checkpoint_model)
            checkpoint_model = rename_state_dict(
                checkpoint_model, self.cfg.finetune.rename_config
            )
            lora_config = {  # specify which layers to add lora to, by default only add to linear layers
                nn.Linear: {
                    "weight": partial(LoRAParametrization.from_linear, rank=8),
                },
                nn.Conv2d: {
                    "weight": partial(LoRAParametrization.from_conv2d, rank=4),
                },
            }
            if (
                any("lora" in k for k in checkpoint_model.keys())
                and self.cfg.finetune.use_lora
            ):
                add_lora_by_name(model, self.cfg.finetune.layers_with_lora, lora_config)
                load_state_dict(
                    model, checkpoint_model, strict=self.cfg.finetune.strict
                )
            elif (
                any("lora" in k for k in checkpoint_model.keys())
                and not self.cfg.finetune.use_lora
            ):
                raise ValueError(
                    "Model checkpoint contains LoRA parameters but use_lora is set to False"
                )
            elif (
                not any("lora" in k for k in checkpoint_model.keys())
                and self.cfg.finetune.use_lora
            ):
                logging.info(
                    "Model checkpoint does not contain LoRA parameters but use_lora is set to True, using the checkpoint as base model"
                )
                load_state_dict(
                    model, checkpoint_model, strict=self.cfg.finetune.strict
                )
                add_lora_by_name(model, self.cfg.finetune.layers_with_lora, lora_config)
            else:
                load_state_dict(
                    model, checkpoint_model, strict=self.cfg.finetune.strict
                )

        # Load additional checkpoints
        if len(self.cfg.finetune.additional_checkpoints) > 0:
            for checkpoint_config in self.cfg.finetune.additional_checkpoints:
                checkpoint_model = load_checkpoint(
                    checkpoint_config.checkpoint, model_key=checkpoint_config.model_key
                )
                checkpoint_model = extract_state_dict(checkpoint_model)
                checkpoint_model = rename_state_dict(
                    checkpoint_model, checkpoint_config.rename_config
                )
                load_state_dict(
                    model, checkpoint_model, strict=checkpoint_config.strict
                )

        if self.cfg.finetune.use_lora:
            # Load LoRA parameters based on the stage
            if self.cfg.stage == "fit":
                # Load LoRA parameters for training
                if self.cfg.finetune.lora_checkpoint is not None:
                    lora_state_dict = load_checkpoint(self.cfg.finetune.lora_checkpoint)
                    lora_state_dict = extract_state_dict(lora_state_dict)
                    lora_state_dict = rename_state_dict(
                        lora_state_dict, self.cfg.finetune.lora_rename_config
                    )
                    load_state_dict(model, lora_state_dict, strict=True)
            elif self.cfg.stage in ["validate", "predict"]:
                # Load LoRA parameters for validation and prediction
                if self.cfg.finetune.lora_checkpoint is not None:
                    lora_state_dict = load_checkpoint(self.cfg.finetune.lora_checkpoint)
                    lora_state_dict = extract_state_dict(lora_state_dict)
                    lora_state_dict = rename_state_dict(
                        lora_state_dict, self.cfg.finetune.lora_rename_config
                    )
                    load_state_dict(model, lora_state_dict, strict=True)

        model.freeze_layers(
            patterns_to_freeze=self.cfg.finetune.patterns_to_freeze, invert_match=False
        )
        logging.debug("Model = %s" % str(model))
        return model

    def on_validation_epoch_end(self):
        pass


class RegionZarrDataModule(RegionDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def build_training_dataset(self, is_train=True):
        return RegionMotifDataset(**self.cfg.dataset, is_train=is_train)

    def build_inference_dataset(self, is_train=False, gene_list=None, gencode_obj=None):
        if gencode_obj is None:
            gencode_obj = get_gencode_obj(self.cfg.assembly)
        logging.debug(gencode_obj)
        return InferenceRegionMotifDataset(
            **self.cfg.dataset,
            assembly=self.cfg.assembly,
            is_train=is_train,
            gene_list=self.cfg.task.gene_list if gene_list is None else gene_list,
            gencode_obj=gencode_obj,
        )


def run(cfg: DictConfig):
    model = RegionLitModel(cfg)
    logging.debug(OmegaConf.to_yaml(cfg))
    dm = RegionDataModule(cfg)
    model.dm = dm

    return run_shared(cfg, model, dm)


def run_zarr(cfg: DictConfig):
    model = RegionLitModel(cfg)
    logging.debug(OmegaConf.to_yaml(cfg))
    dm = RegionZarrDataModule(cfg)
    model.dm = dm

    return run_shared(cfg, model, dm)
