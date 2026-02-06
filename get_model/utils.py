# --------------------------------------------------------
# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import logging
import math
import os

import hydra
import numpy as np
import torch
import torch.utils.data
import zarr
from hydra.core.global_hydra import GlobalHydra

np.bool = np.bool_

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.plugins import MixedPrecision
from omegaconf import OmegaConf


def setup_wandb(cfg):
    wandb_logger = WandbLogger(
        name=cfg.run.run_name,
        project=cfg.run.project_name,
        save_dir=os.path.join(
            cfg.machine.output_dir, cfg.run.project_name, cfg.run.run_name
        ),
    )
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    return wandb_logger


def setup_trainer(cfg):
    if cfg.machine.num_devices > 0:
        strategy = "auto"
        accelerator = "gpu" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        device = cfg.machine.num_devices
        if cfg.machine.num_devices > 1:
            strategy = "ddp_find_unused_parameters_true"
    else:
        strategy = "auto"
        accelerator = "cpu"
        device = "auto"

    # create output dir if not exist
    os.makedirs(
        os.path.join(cfg.machine.output_dir, cfg.run.project_name, cfg.run.run_name),
        exist_ok=True,
    )
    logger = []
    if cfg.run.use_wandb:
        wandb_logger = setup_wandb(cfg)
        logger.append(wandb_logger)
    logger.append(
        CSVLogger(
            save_dir=os.path.join(
                cfg.machine.output_dir,
                cfg.run.project_name,
                cfg.run.run_name,
                "csv_logs",
            )
        )
    )
    # Create both regular and finetuned checkpoints
    save_top_k = -1 if cfg.training.save_every_n_epochs is not None else 1
    filename = "checkpoint-{epoch:03d}-{step:06d}-{val_loss:.4f}" if cfg.training.save_every_n_epochs is not None else "best"
    regular_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=save_top_k,
        save_last=True,
        every_n_epochs=cfg.training.save_every_n_epochs,
        filename=filename,
        dirpath=os.path.join(
            cfg.machine.output_dir,
            cfg.run.project_name,
            cfg.run.run_name,
            "checkpoints",
        ),
    )

    callbacks = [regular_checkpoint]

    # Add LearningRateMonitor if needed
    if cfg.training.get("add_lr_monitor", False):
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    # Determine inference mode
    inference_mode = True
    if "interpret" in cfg.task.test_mode:
        inference_mode = False
    plugins = []
    if cfg.training.use_fp16:
        plugins.append(MixedPrecision(precision="16-mixed", device="cuda"))
    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=accelerator,
        num_sanity_val_steps=10,
        strategy=strategy,
        devices=device,
        logger=logger,
        callbacks=callbacks,
        plugins=plugins,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        gradient_clip_val=cfg.training.clip_grad,
        log_every_n_steps=cfg.training.log_every_n_steps,
        val_check_interval=cfg.training.val_check_interval,
        inference_mode=inference_mode,
        default_root_dir=os.path.join(
            cfg.machine.output_dir, cfg.run.project_name, cfg.run.run_name
        ),
    )

    return trainer


def load_config(config_name):
    # Initialize Hydra to load the configuration
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="config", version_base="1.3")
    cfg = hydra.compose(config_name=config_name)
    return cfg


def print_shape(x):
    """a recursive function to print the shape of values in a nested dictionary"""
    if isinstance(x, dict):
        for k, v in x.items():
            logging.debug(k, end=": ")
            print_shape(v)
    elif isinstance(x, np.ndarray) or isinstance(x, torch.Tensor):
        logging.debug(x.shape)
    elif isinstance(x, list):
        logging.debug(len(x))
    else:
        logging.debug(x)


def load_checkpoint(checkpoint_path, model_key=None):
    if checkpoint_path.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_path, map_location="cpu", check_hash=True
        )
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print("Load ckpt from %s" % checkpoint_path)

    if model_key is not None:
        checkpoint_model = None
        for key in model_key.split("|"):
            if key in checkpoint:
                checkpoint_model = checkpoint[key]
                print("Load state_dict by model_key = %s" % key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
    else:
        checkpoint_model = checkpoint

    return checkpoint_model


def rename_lit_state_dict(state_dict, patterns_to_drop=[]):
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace("model.", "")
        new_state_dict[new_key] = state_dict[key]
        for key_to_drop in patterns_to_drop:
            if key_to_drop in new_key:
                del new_state_dict[new_key]
    return new_state_dict


def extract_state_dict(checkpoint_model):
    if "model" in checkpoint_model:
        checkpoint_model = checkpoint_model["model"]
    if "state_dict" in checkpoint_model:
        checkpoint_model = checkpoint_model["state_dict"]
    return checkpoint_model


def rename_state_dict(state_dict, rename_config):
    if rename_config is None:
        return state_dict

    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        for pattern, replacement in rename_config.items():
            new_key = new_key.replace(pattern, replacement)
        new_state_dict[new_key] = value

    return new_state_dict


def freeze_layers(model, freeze_last_layer=False, freeze_atac_attention=False):
    if freeze_last_layer:
        for name, param in model.named_parameters():
            if not (
                name.startswith("blocks.11")
                or name.startswith("head.")
                or name.startswith("fc_norm")
                or name.startswith("norm")
            ):
                logging.debug(name)
                param.requires_grad = False

    if freeze_atac_attention:
        for name, param in model.named_parameters():
            if "atac_attention" in name:
                param.requires_grad = False
                print(f"Freezed weights of {name}")


def load_state_dict(model, state_dict, strict=True, patterns_to_drop=[]):
    # Remove keys matching the patterns_to_drop
    for pattern in patterns_to_drop:
        state_dict = {k: v for k, v in state_dict.items() if pattern not in k}
    model.load_state_dict(state_dict, strict=strict)


def recursive_detach(tensors):
    if isinstance(tensors, dict):
        return {k: recursive_detach(v) for k, v in tensors.items()}
    elif isinstance(tensors, list):
        return [recursive_detach(v) for v in tensors]
    elif isinstance(tensors, torch.Tensor):
        if tensors.is_cuda or tensors.device.type == "mps":
            return tensors.detach().cpu()
        else:
            return tensors.detach()
    else:
        return tensors


def recursive_numpy(tensors, dtype=None):
    if isinstance(tensors, dict):
        return {k: recursive_numpy(v, dtype) for k, v in tensors.items()}
    elif isinstance(tensors, list):
        return [recursive_numpy(v, dtype) for v in tensors]
    elif isinstance(tensors, torch.Tensor):
        if dtype is None:
            return tensors.detach().cpu().float().numpy()
        else:
            return tensors.detach().cpu().float().numpy().astype(dtype)
    else:
        return tensors


def recursive_concat_numpy(list_of_dict):
    """input is a list of dict, each dict has the same keys, certain keys might corresponding to another dict,
    then concatenate the values of the same key in the list of dict, return a dict, recursively handle the hierarchical structure
    """
    if isinstance(list_of_dict[0], dict):
        keys = list_of_dict[0].keys()
        return {k: recursive_concat_numpy([d[k] for d in list_of_dict]) for k in keys}
    elif isinstance(list_of_dict[0], list):
        return [
            recursive_concat_numpy([d[i] for d in list_of_dict])
            for i in range(len(list_of_dict[0]))
        ]
    elif isinstance(list_of_dict[0], np.ndarray):
        # Handle string arrays (like gene names and chromosomes) differently
        if list_of_dict[0].dtype.kind in ['U', 'S', 'O']:
            # Flatten the array if it's multi-dimensional
            flattened_arrays = [arr.flatten() if arr.ndim > 1 else arr for arr in list_of_dict]
            return np.concatenate(flattened_arrays)
        return np.concatenate(list_of_dict, axis=0)
    else:
        return list_of_dict


def recursive_save_to_zarr(zarr_group, dict_data, **kwargs):
    from numcodecs import Blosc
    for k, v in dict_data.items():
        if isinstance(v, dict):
            subgroup = zarr_group.require_group(k)
            recursive_save_to_zarr(subgroup, v, **kwargs)
        else:
            # if group not exist, create it
            if k not in zarr_group:
                # respect the dtype of the data
                if isinstance(v, np.ndarray):
                    zarr_group.create_dataset(k, data=v, dtype=v.dtype, compressor=Blosc(cname='zstd', clevel=3, shuffle=1), **kwargs)
                else:
                    zarr_group.create_dataset(k, data=v, compressor=Blosc(cname='zstd', clevel=3, shuffle=1), **kwargs)
            else:  # append to existing group
                # pad to the same shape
                if (
                    isinstance(v, np.ndarray)
                    and isinstance(zarr_group[k], zarr.Array)
                    and zarr_group[k].shape[1:] != v.shape[1:]
                ):
                    # Handle 1D arrays differently
                    if v.ndim == 1:
                        zarr_group[k].append(v)
                    else:
                        new_shape = [v.shape[0]] + list(zarr_group[k].shape[1:])
                        new_data = np.zeros(new_shape, dtype=v.dtype)
                        new_data[:, : v.shape[1]] = v
                        zarr_group[k].append(new_data)
                else:
                    zarr_group[k].append(v)


def cosine_scheduler(
    base_value,
    final_value,
    epochs,
    niter_per_ep,
    warmup_epochs=0,
    start_warmup_value=0,
    warmup_steps=-1,
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    logging.debug("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [
            final_value
            + 0.5
            * (base_value - final_value)
            * (1 + math.cos(math.pi * i / (len(iters))))
            for i in iters
        ]
    )

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule

def recursive_print_shape(zarr_path, prefix=''):
    z = zarr.open(zarr_path)
    for key, value in z.items():
        if isinstance(value, zarr.Array):
            print(f"{prefix}/{key}: {value.shape}")
        else:
            recursive_print_shape(value, f"{prefix}/{key}")