# --------------------------------------------------------
# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import io
import math

import hydra
import numpy as np
import torch
import torch.utils.data
import zarr
from hydra.core.global_hydra import GlobalHydra

np.bool = np.bool_


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
            print(k)
            print_shape(v)
    elif isinstance(x, np.ndarray) or isinstance(x, torch.Tensor):
        print(x.shape)
    elif isinstance(x, list):
        print(len(x))
    else:
        print(x)


def load_checkpoint(checkpoint_path, model_key=None):
    if checkpoint_path.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(checkpoint_path, map_location="cpu", check_hash=True)
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
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

def remove_keys(checkpoint_model, model_state_dict):
    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != model_state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]



def rename_keys(state_dict):
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key
        if "blocks." in new_key:
            new_key = new_key.replace("blocks.", "encoder.blocks.")
        if "fc_norm." in new_key:
            new_key = new_key.replace("fc_norm.", "encoder.norm.")
        if "head." in new_key:
            new_key = new_key.replace("head.", "head_exp.head.")
        if "region_embed.proj." in new_key:
            new_key = new_key.replace(
                "region_embed.proj.", "region_embed.embed.")
        new_state_dict[new_key] = state_dict[key]

    if 'region_embed.embed.weight' in new_state_dict:
        # .unsqueeze(2)
        new_state_dict['region_embed.embed.weight'] = new_state_dict['region_embed.embed.weight']
    return new_state_dict


def rename_lit_state_dict(state_dict, patterns_to_drop=[]):
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace("model.", "")
        new_state_dict[new_key] = state_dict[key]
        for key_to_drop in patterns_to_drop:
            if key_to_drop in new_key:
                del new_state_dict[new_key]
    return new_state_dict


def rename_v1_pretrain_keys(state_dict):
    """
    Rename the keys in the state dictionary.
    """
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace("encoder.head.", "head_mask.")
        new_key = new_key.replace(
            "encoder.region_embed", "region_embed")
        new_key = new_key.replace(
            "region_embed.proj.", "region_embed.embed.")
        new_key = new_key.replace(
            "encoder.cls_token", "cls_token")

        new_state_dict[new_key] = state_dict[key]
    return new_state_dict


def rename_v1_finetune_keys(state_dict):
    """
    Rename the keys in the state dictionary.
    """
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace("blocks.", "encoder.blocks.")
        new_key = new_key.replace("fc_norm.", "encoder.norm.")
        new_key = new_key.replace("encoder.head.", "head_mask.")
        new_key = new_key.replace(
            "encoder.region_embed", "region_embed")
        new_key = new_key.replace(
            "region_embed.proj.", "region_embed.embed.")
        new_key = new_key.replace(
            "encoder.cls_token", "cls_token")
        new_key = new_key.replace(
            "head.", "head_exp.head.")

        new_state_dict[new_key] = state_dict[key]
    return new_state_dict


def extract_state_dict(checkpoint_model):
    if 'model' in checkpoint_model:
        checkpoint_model = checkpoint_model['model']
    if 'state_dict' in checkpoint_model:
        checkpoint_model = checkpoint_model['state_dict']
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
                print(name)
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
        if tensors.is_cuda:
            return tensors.detach().cpu()
    else:
        return tensors

def recursive_numpy(tensors):
    if isinstance(tensors, dict):
        return {k: recursive_numpy(v) for k, v in tensors.items()}
    elif isinstance(tensors, list):
        return [recursive_numpy(v) for v in tensors]
    elif isinstance(tensors, torch.Tensor):
        return tensors.detach().cpu().float().numpy()
    else:
        return tensors
    
def recursive_concat_numpy(list_of_dict):
    """input is a list of dict, each dict has the same keys, certain keys might corresponding to another dict,
    then concatenate the values of the same key in the list of dict, return a dict, recursively handle the hierarchical structure"""
    if isinstance(list_of_dict[0], dict):
        keys = list_of_dict[0].keys()
        return {k: recursive_concat_numpy([d[k] for d in list_of_dict]) for k in keys}
    elif isinstance(list_of_dict[0], list):
        return [recursive_concat_numpy([d[i] for d in list_of_dict]) for i in range(len(list_of_dict[0]))]
    elif isinstance(list_of_dict[0], np.ndarray):
        return np.concatenate(list_of_dict, axis=0)
    else:
        return list_of_dict
    
def recursive_save_to_zarr(zarr_group, dict_data, **kwargs):
    for k, v in dict_data.items():
        if isinstance(v, dict):
            subgroup = zarr_group.require_group(k)
            recursive_save_to_zarr(subgroup, v, **kwargs)
        else:
            # if group not exist, create it
            if k not in zarr_group:
                zarr_group.create_dataset(k, data=v, **kwargs)
            else: # append to existing group
                # pad to the same shape
                if isinstance(v, np.ndarray) and isinstance(zarr_group[k], zarr.core.Array) and zarr_group[k].shape[1:] != v.shape[1:]:
                    new_shape = [v.shape[0]] + list(zarr_group[k].shape[1:])
                    new_data = np.zeros(new_shape, dtype=v.dtype)
                    new_data[:, :v.shape[1]] = v
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
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(
            start_warmup_value, base_value, warmup_iters)
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
