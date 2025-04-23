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
import zarr
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from minlora import LoRAParametrization
from minlora.model import add_lora_by_name
from omegaconf import DictConfig
from tqdm import tqdm

from get_model.config.config import *
from get_model.dataset.collate import get_perturb_collate_fn, get_rev_collate_fn

try:
    from get_model.dataset.zarr_dataset import (
        InferenceDataset,
        PerturbationInferenceDataset,
        PretrainDataset,
        get_gencode_obj,
        get_sequence_obj,
    )
except:
    pass
from get_model.model.model import *
from get_model.model.modules import *
from get_model.optim import LayerDecayValueAssigner, create_optimizer
from get_model.utils import (
    cosine_scheduler,
    extract_state_dict,
    load_checkpoint,
    load_state_dict,
    recursive_concat_numpy,
    recursive_detach,
    recursive_numpy,
    recursive_save_to_zarr,
    rename_state_dict,
    setup_trainer,
)

logging.disable(logging.WARN)


# Wrapper model for Captum
class WrapperModel(torch.nn.Module):
    def __init__(self, model, focus, output_key=None):
        super(WrapperModel, self).__init__()
        self.model = model
        self.focus = focus
        self.output_key = output_key

    def forward(self, input, strand, *args, **kwargs):
        output = self.model.forward(input)
        if isinstance(output, dict):
            logging.debug("output is dict with keys" + str(output.keys()))
            return output[self.output_key][:, self.focus, strand]
        return output[:, self.focus, strand]


def extract_peak_df(batch):
    logging.debug(list(batch.keys()))
    peak_coord = batch["peak_coord"][0].cpu().numpy()
    chr_name = batch["chromosome"][0]
    df = pd.DataFrame(peak_coord, columns=["Start", "End"])
    df["Chromosome"] = chr_name
    return df[["Chromosome", "Start", "End"]]


def get_insulation_overlap(batch, insulation):
    from pyranges import PyRanges as pr

    peak_df = extract_peak_df(batch)
    tss_peak = int(batch["tss_peak"][0].cpu())
    insulation = insulation[insulation["Chromosome"] == peak_df["Chromosome"].values[0]]
    logging.debug(pr(peak_df.iloc[tss_peak : tss_peak + 1]))
    overlap = (
        pr(peak_df.iloc[tss_peak : tss_peak + 1])
        .join(pr(insulation), suffix="_insulation")
        .df
    )
    final_insulation = (
        overlap.sort_values("mean_num_celltype")
        .iloc[-1][["Chromosome", "Start_insulation", "End_insulation"]]
        .rename({"Start_insulation": "Start", "End_insulation": "End"})
    )
    subset_peak_df = peak_df.loc[
        (peak_df.Start > final_insulation.Start) & (peak_df.End < final_insulation.End)
    ]
    new_peak_start_idx = subset_peak_df.index.min()
    new_peak_end_idx = subset_peak_df.index.max()
    new_tss_peak = tss_peak - new_peak_start_idx
    return new_peak_start_idx, new_peak_end_idx, new_tss_peak


class LitModel(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = self.get_model()
        self.loss = self.model.loss
        self.metrics = self.model.metrics
        self.lr = cfg.optimizer.lr
        self.save_hyperparameters()
        self.dm = None
        self.accumulated_results = []

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

    def forward(self, batch):
        return self.model(**batch)

    def _shared_step(self, batch, batch_idx, stage="train"):
        input = self.model.get_input(batch)
        output = self(input)
        pred, obs = self.model.before_loss(output, batch)
        loss = self.loss(pred, obs)
        # if loss is a dict, rename the keys with the stage prefix
        distributed = self.cfg.machine.num_devices > 1
        if stage != "predict":
            if isinstance(loss, dict):
                loss = {f"{stage}_{key}": value for key, value in loss.items()}
                self.log_dict(
                    loss, batch_size=self.cfg.machine.batch_size, sync_dist=distributed
                )
            loss = self.model.after_loss(loss)
        return loss, pred, obs

    def training_step(self, batch, batch_idx):
        loss, pred, obs = self._shared_step(batch, batch_idx, stage="train")
        self.log(
            "train_loss",
            loss,
            batch_size=self.cfg.machine.batch_size,
            sync_dist=self.cfg.machine.num_devices > 1,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, obs = self._shared_step(batch, batch_idx, stage="val")
        metrics = self.metrics(pred, obs)
        self.log_dict(
            metrics,
            batch_size=self.cfg.machine.batch_size,
            sync_dist=self.cfg.machine.num_devices > 1,
        )
        self.log(
            "val_loss",
            loss,
            batch_size=self.cfg.machine.batch_size,
            sync_dist=self.cfg.machine.num_devices > 1,
        )
        # log the best metric across epoch

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

    def test_step(self, batch, batch_idx):
        loss, pred, obs = self._shared_step(batch, batch_idx, stage="test")
        metrics = self.metrics(pred, obs)
        self.log_dict(
            metrics,
            batch_size=self.cfg.machine.batch_size,
            sync_dist=self.cfg.machine.num_devices > 1,
        )
        self.log(
            "test_loss",
            loss,
            batch_size=self.cfg.machine.batch_size,
            sync_dist=self.cfg.machine.num_devices > 1,
        )
        return pred, obs

    def predict_step(self, batch, batch_idx, *args, **kwargs):
        if self.cfg.task.test_mode == "perturb":
            return self.perturb_step(batch, batch_idx)
        elif self.cfg.task.test_mode == "predict":
            loss, pred, obs = self._shared_step(batch, batch_idx, stage="predict")
            return pred, obs
        elif self.cfg.task.test_mode == "interpret":
            # assume focus is the center peaks in the input sample
            with torch.enable_grad():
                result = self.interpret_step(
                    batch,
                    batch_idx,
                    layer_names=self.cfg.task.layer_names,
                    focus=self.cfg.dataset.n_peaks_upper_bound // 2,
                )
                self.accumulated_results.append(result)
                
        elif self.cfg.task.test_mode == "perturb_interpret":
            with torch.enable_grad():
                return self.perturb_interpret_step(batch, batch_idx)

    def perturb_step(self, batch, batch_idx):
        """Perturb the input sequence and do inference on both."""
        batch_wt = batch["WT"]
        batch_mut = batch["MUT"]

        input_wt = self.model.get_input(batch_wt, perturb=True)
        output_wt = self(input_wt)
        input_mut = self.model.get_input(batch_mut, perturb=True)
        output_mut = self(input_mut)
        pred_wt, obs_wt = self.model.before_loss(output_wt, batch_wt)
        pred_mut, obs_mut = self.model.before_loss(output_mut, batch_mut)
        return {
            "pred_wt": pred_wt,
            "obs_wt": obs_wt,
            "pred_mut": pred_mut,
            "obs_mut": obs_mut,
        }

    def perturb_interpret_step(self, batch, batch_idx):
        """Perturb the input sequence and do interpretation on both."""
        batch_wt = batch["WT"]
        batch_mut = batch["MUT"]

        pred_wt, obs_wt, jacobians_wt, embeddings_wt = self.interpret_step(
            batch_wt,
            batch_idx,
            layer_names=self.cfg.task.layer_names,
            focus=self.cfg.dataset.n_peaks_upper_bound // 2,
        )
        pred_mut, obs_mut, jacobians_mut, embeddings_mut = self.interpret_step(
            batch_mut,
            batch_idx,
            layer_names=self.cfg.task.layer_names,
            focus=self.cfg.dataset.n_peaks_upper_bound // 2,
        )
        return {
            "pred_wt": pred_wt,
            "obs_wt": obs_wt,
            "pred_mut": pred_mut,
            "obs_mut": obs_mut,
            "jacobians_wt": jacobians_wt,
            "embeddings_wt": embeddings_wt,
            "jacobians_mut": jacobians_mut,
            "embeddings_mut": embeddings_mut,
        }

    def interpret_step(self, batch, batch_idx, layer_names: List[str] = None, focus: int | np.ndarray = None):
        target_tensors = {}
        hooks = []
        input = self.model.get_input(batch)
        assert focus is not None, "Please provide a focus position for interpretation"
        assert layer_names is not None, "Please provide a list of layer names for interpretation"
        
        # Register hooks to capture target tensors
        def capture_target_tensor(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0] # get the first element of the tuple, usually x rather than (x, attn)
                output.retain_grad()
                if name == "region_embed":
                    target_tensors[name] = output
                elif 'encoder' in name: # in encoder, a cls token is added in the first position
                    target_tensors[name] = output[:, 1+focus, :]
                else:
                    target_tensors[name] = output[:, focus, :]
            return hook

        # Setup hooks and input tensors
        if layer_names is None or len(layer_names) == 0:
            target_tensors["input"] = input
            for key, tensor in input.items():
                tensor.requires_grad = True
        else:
            target_tensors["input"] = input
            for key, tensor in input.items():
                tensor.requires_grad = True
            for layer_name in layer_names:
                layer = self.model.get_submodule(layer_name)
                hook = layer.register_forward_hook(capture_target_tensor(layer_name))
                hooks.append(hook)

        # Initial forward pass
        output = self(input)
        pred_original, obs_original = self.model.before_loss(output, batch)

        # Compute jacobians for each target and strand
        jacobians = {}
        for target_name, target in obs_original.items():
            jacobians[target_name] = {}
            for i in range(target.shape[-1]):  # Loop over strands
                # Fresh forward pass for each strand
                self.zero_grad(set_to_none=True)
                output = self(input)
                pred, obs = self.model.before_loss(output, batch)
                
                # Create mask for current strand
                mask = torch.zeros_like(pred[target_name]).to(self.device)
                if isinstance(focus, int):
                    mask[:, focus, i] = 1
                else:
                    assert len(focus) == mask.shape[0]
                    for j in range(mask.shape[0]):
                        mask[j, focus[j], i] = 1
                
                # Backward pass
                pred[target_name].backward(mask)
                
                # Store gradients
                jacobians[target_name][str(i)] = {}
                for layer_name, layer in target_tensors.items():
                    if isinstance(layer, torch.Tensor):
                        if layer.grad is not None:
                            jacobians[target_name][str(i)][layer_name] = layer.grad.detach().cpu().numpy()
                    elif isinstance(layer, dict):
                        jacobians[target_name][str(i)][layer_name] = {}
                        for layer_input_name, layer_input in layer.items():
                            if layer_input.grad is not None:
                                jacobians[target_name][str(i)][layer_name][layer_input_name] = \
                                    layer_input.grad.detach().cpu().numpy()

        # Clean up hooks
        for hook in hooks:
            hook.remove()

        # Prepare return values
        pred = recursive_numpy(recursive_detach(pred_original))
        obs = recursive_numpy(recursive_detach(obs_original))
        jacobians = recursive_numpy(jacobians, dtype=np.float16)
        target_tensors = recursive_numpy(target_tensors, dtype=np.float16)
        return pred, obs, jacobians, target_tensors

    def _save_accumulated_results(self):
        """Helper method to save accumulated results to zarr and clear the list"""
        if not self.accumulated_results:
            return
            
        zarr_path = f"{self.cfg.machine.output_dir}/{self.cfg.run.project_name}/{self.cfg.run.run_name}/{self.cfg.dataset.leave_out_celltypes}.zarr"
        print(f"Saving batch of results to {zarr_path}")

        from numcodecs import VLenUTF8

        object_codec = VLenUTF8()
        z = zarr.open(zarr_path, mode="a")
        print(self.accumulated_results[0].keys())
        # Concatenate all accumulated results
        accumulated_results = recursive_concat_numpy(self.accumulated_results)
        print(accumulated_results.keys())
        
        # Ensure gene names and chromosomes are properly formatted as string arrays
        if 'available_genes' in accumulated_results:
            accumulated_results['available_genes'] = np.array(accumulated_results['available_genes'], dtype='U100')
        if 'chromosome' in accumulated_results:
            accumulated_results['chromosome'] = np.array(accumulated_results['chromosome'], dtype='U30')

        recursive_save_to_zarr(
            z, accumulated_results, object_codec=object_codec, overwrite=False
        )

        # Clear accumulated results and force garbage collection
        self.accumulated_results = []
        gc.collect()

    def on_predict_epoch_end(self):
        if self.cfg.task.test_mode == "interpret":
            # Save any remaining accumulated results
            self._save_accumulated_results()

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if self.cfg.task.test_mode == "interpret":
            if len(self.accumulated_results) >= 100:
                self._save_accumulated_results()
            
    def configure_optimizers(self):
        if hasattr(self.model.cfg, "encoder"):
            num_layers = self.model.cfg.encoder.num_layers
        else:
            num_layers = 0
        assigner = LayerDecayValueAssigner(
            list(0.75 ** (num_layers + 1 - i) for i in range(num_layers + 2))
        )

        if assigner is not None:
            logging.debug("Assigned values = %s" % str(assigner.values))
        skip_weight_decay_list = self.model.no_weight_decay()
        logging.debug("Skip weight decay list: %s" % str(skip_weight_decay_list))

        optimizer = create_optimizer(
            self.cfg.optimizer,
            self.model,
            skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None,
        )

        data_size = len(self.dm.dataset_train)
        num_gpu_or_cpu_devices = (
            self.cfg.machine.num_devices if self.cfg.machine.num_devices > 0 else 1
        )
        num_training_steps_per_epoch = (
            data_size // self.cfg.machine.batch_size // num_gpu_or_cpu_devices
        )
        self.schedule = cosine_scheduler(
            base_value=self.cfg.optimizer.lr,
            final_value=self.cfg.optimizer.min_lr,
            epochs=self.cfg.training.epochs,
            niter_per_ep=num_training_steps_per_epoch,
            warmup_epochs=self.cfg.training.warmup_epochs,
            start_warmup_value=self.cfg.optimizer.min_lr,
            warmup_steps=-1,
        )
        # Suppose self.schedule is a per-step LR array of length = total training steps.
        # We define a function that picks the LR from self.schedule by the current "step".
        def lr_lambda(step_index):
            # step_index will go from 0 ... (max_steps - 1)
            # (Lightning calls scheduler.step() after each batch if interval="step")
            # if self.schedule[step_index] is the actual LR value, we divide by initial lr
            # because LambdaLR multiplies the optimizer’s base LR by this factor
            return self.schedule[step_index-1] / self.cfg.optimizer.lr

        # Create the LambdaLR
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda,
        )

        # Let Lightning know: this scheduler is stepped each batch
        scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",  # call scheduler.step() every batch
            "frequency": 1,
        }

        return [optimizer], [scheduler_config]


    # def on_before_optimizer_step(self, optimizer):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     # norms = grad_norm(self.model, norm_type=2)
    #     # self.log_dict(norms)

    # def on_validation_epoch_end(self):
    #     if self.cfg.dataset_name != 'bias_thp1':
    #         trainer = self.trainer
    #         metric = run_ppif_task(trainer, self)
    #         step = trainer.global_step
    #         self.logger.log_metrics(metric, step)

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

    def _shared_build_dataset(self, is_train, sequence_obj=None):
        # config = self.cfg.dataset.dataset_configs[self.cfg.dataset_name]
        # merge config with self.cfg.dataset
        config = DictConfig({**self.cfg.dataset})

        root = self.cfg.machine.data_path
        codebase = self.cfg.machine.codebase
        assembly = self.cfg.assembly
        config.dataset_size = (
            config.dataset_size if is_train else config.eval_dataset_size
        )
        config.zarr_dirs = [f"{root}/{zarr_dir}" for zarr_dir in config.zarr_dirs]
        config.genome_seq_zarr = f"{root}/{assembly}.zarr"
        config.genome_motif_zarr = f"{root}/{assembly}_motif_result.zarr"
        config.insulation_paths = [
            f"{codebase}/data/{assembly}_4DN_average_insulation.ctcf.adjecent.feather",
            f"{codebase}/data/{assembly}_4DN_average_insulation.ctcf.longrange.feather",
        ]
        self.dataset_config = config
        sequence_obj = (
            sequence_obj
            if sequence_obj is not None
            else get_sequence_obj(config.genome_seq_zarr)
        )
        gencode_obj = get_gencode_obj(config.genome_seq_zarr, root)

        return config, sequence_obj, gencode_obj

    def build_training_dataset(self, sequence_obj, is_train=True) -> None:
        config, sequence_obj, gencode_obj = self._shared_build_dataset(
            is_train, sequence_obj=sequence_obj
        )

        # Create dataset with configuration parameters
        dataset = PretrainDataset(
            is_train=is_train, sequence_obj=sequence_obj, **config
        )

        return dataset

    def build_inference_dataset(self, sequence_obj, gene_list=None):
        config, sequence_obj, gencode_obj = self._shared_build_dataset(
            is_train=False, sequence_obj=sequence_obj
        )
        if hasattr(self, "mutations") and self.mutations is not None:
            mutations = self.mutations
        else:
            mutations = config["mutations"]
        config.pop("mutations", None)
        # no need to leave out chromosomes or celltypes in inference
        # config['leave_out_chromosomes'] = ""
        config["random_shift_peak"] = None
        dataset = InferenceDataset(
            is_train=False,
            assembly=self.cfg.assembly,
            gencode_obj=gencode_obj,
            gene_list=self.cfg.task.gene_list if gene_list is None else gene_list,
            mutations=mutations,
            sequence_obj=sequence_obj,
            **config,
        )
        return dataset

    def build_perturb_dataset(
        self, perturbations, perturb_mode, sequence_obj, gene_list=None
    ):
        inference_dataset = self.build_inference_dataset(
            sequence_obj, gene_list=gene_list
        )
        dataset = PerturbationInferenceDataset(
            inference_dataset, perturbations, perturb_mode
        )
        return dataset

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        genome_seq_zarr = f"{self.cfg.machine.data_path}/{self.cfg.assembly}.zarr"
        sequence_obj = get_sequence_obj(genome_seq_zarr)

        if stage == "fit" or stage is None:
            self.dataset_train = self.build_training_dataset(
                sequence_obj=sequence_obj, is_train=True
            )
            self.dataset_val = self.build_training_dataset(
                sequence_obj=sequence_obj, is_train=False
            )
        if stage == "predict":
            if self.cfg.task.test_mode == "predict":
                self.dataset_predict = self.build_training_dataset(
                    sequence_obj=sequence_obj, is_train=False
                )
            elif (
                self.cfg.task.test_mode == "interpret"
                or self.cfg.task.test_mode == "inference"
                or self.cfg.task.test_mode == "interpret_captum"
            ):
                self.mutations = None
                self.dataset_predict = self.build_inference_dataset(
                    sequence_obj=sequence_obj
                )
            elif "perturb" in self.cfg.task.test_mode:
                self.mutations = pd.read_csv(self.cfg.task.mutations, sep="\t")
                if self.mutations is not None:
                    self.perturbation_mode = "mutation"
                elif self.cfg.dataset.peak_inactivation is not None:
                    self.perturbation_mode = "peak_inactivation"
                self.dataset_predict = self.build_perturb_dataset(
                    perturbations=self.mutations,
                    perturb_mode=self.perturbation_mode,
                    sequence_obj=sequence_obj,
                    gene_list=self.cfg.task.gene_list,
                )

        if stage == "validate":
            self.dataset_val = self.build_training_dataset(
                sequence_obj=sequence_obj, is_train=False
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.cfg.machine.batch_size,
            num_workers=self.cfg.machine.num_workers,
            drop_last=True,
            collate_fn=get_rev_collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.cfg.machine.batch_size,
            num_workers=self.cfg.machine.num_workers,
            drop_last=True,
            collate_fn=get_rev_collate_fn,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=self.cfg.machine.batch_size,
            num_workers=self.cfg.machine.num_workers,
            drop_last=True,
            collate_fn=(
                get_rev_collate_fn
                if "perturb" not in self.cfg.task.test_mode
                else get_perturb_collate_fn
            ),
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_predict,
            batch_size=self.cfg.machine.batch_size,
            num_workers=self.cfg.machine.num_workers,
            drop_last=False,
            collate_fn=(
                get_rev_collate_fn
                if "perturb" not in self.cfg.task.test_mode
                else get_perturb_collate_fn
            ),
        )


def run_shared(cfg, model, dm, trainer=None):
    if trainer is None:
        trainer = setup_trainer(cfg)

    if cfg.stage == "fit":
        trainer.fit(model, dm, ckpt_path=cfg.finetune.resume_ckpt)
    if cfg.stage == "validate":
        trainer.validate(model, datamodule=dm, ckpt_path=cfg.finetune.resume_ckpt)
    if cfg.stage == "predict":
        trainer.predict(model, datamodule=dm, ckpt_path=cfg.finetune.resume_ckpt)
    # close wandb
    wandb.finish()
    return trainer


def run(cfg: DictConfig):
    torch.set_float32_matmul_precision("medium")
    model = LitModel(cfg)
    dm = GETDataModule(cfg)
    model.dm = dm

    return run_shared(cfg, model, dm)


def run_downstream(cfg: DictConfig):
    torch.set_float32_matmul_precision("medium")
    # if cfg.finetune.checkpoint is not None:
    # model = LitModel.load_from_checkpoint(cfg.finetune.checkpoint)
    # else:
    model = LitModel(cfg)
    # move the model to the gpu
    model.to("cuda")
    dm = GETDataModule(cfg)
    model.dm = dm
    trainer = setup_trainer(cfg)
    logging.debug(run_ppif_task(trainer, model))


def run_ppif_task(trainer: L.Trainer, lm: LitModel, output_key="atpm"):
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    mutation = pd.read_csv(lm.cfg.task.mutations, sep="\t")
    n_mutation = mutation.shape[0]
    n_peaks_upper_bound = lm.cfg.dataset.n_peaks_upper_bound
    result = []
    # setup dataset_predict
    lm.dm.setup(stage="predict")
    with torch.no_grad():
        lm.to("cuda")
        for i, batch in tqdm(
            enumerate(lm.dm.predict_dataloader()), total=len(lm.dm.predict_dataloader())
        ):
            batch = lm.transfer_batch_to_device(batch, lm.device, dataloader_idx=0)
            out = lm.predict_step(batch, i)
            result.append(out)
    pred_wt = [r["pred_wt"][output_key] for r in result]
    pred_mut = [r["pred_mut"][output_key] for r in result]
    n_celltypes = lm.dm.dataset_predict.inference_dataset.datapool.n_celltypes
    pred_wt = torch.cat(pred_wt, dim=0).reshape(
        n_celltypes, n_mutation, n_peaks_upper_bound
    )[0, :, n_peaks_upper_bound // 2]
    pred_mut = torch.cat(pred_mut, dim=0).reshape(
        n_celltypes, n_mutation, n_peaks_upper_bound
    )[0, :, n_peaks_upper_bound // 2]
    pred_change = (10**pred_mut - 10**pred_wt) / (10**pred_wt - 1) * 100
    mutation["pred_change"] = pred_change.detach().cpu().numpy()
    y = (
        mutation.query("`corrected p value`<=0.05")
        .query('Screen.str.contains("Pro")')
        .query('Screen.str.contains("Tiling")')["% change to PPIF expression"]
        .values
    )
    x = (
        mutation.query("`corrected p value`<=0.05")
        .query('Screen.str.contains("Pro")')
        .query('Screen.str.contains("Tiling")')["pred_change"]
        .values
    )
    pearson = np.corrcoef(x, y)[0, 1]
    r2 = r2_score(y, x)
    spearman = spearmanr(x, y)[0]
    slope = LinearRegression().fit(x.reshape(-1, 1), y).coef_[0]
    # save a scatterplot
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.scatterplot(x=x, y=y)
    plt.xlabel("Predicted change in PPIF expression")
    plt.ylabel("Observed change in PPIF expression")
    plt.savefig(f"{lm.cfg.machine.output_dir}/ppif_scatterplot.png")
    return {
        "ppif_pearson": pearson,
        "ppif_spearman": spearman,
        "ppif_r2": r2,
        "ppif_slope": slope,
    }



