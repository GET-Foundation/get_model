from tqdm import tqdm
# import hicstraw
from minlora.model import add_lora_by_name
from get_model.utils import rename_lit_state_dict, rename_state_dict
import hydra
from omegaconf import OmegaConf
from get_model.dataset.zarr_dataset import (InferenceDataset,
                                            InferenceReferenceRegionDataset,
                                            ReferenceRegionMotif,
                                            ReferenceRegionMotifConfig)
import torch.utils
from get_model.run_ref_region import *
import random

from caesar.io.gencode import Gencode
from caesar.io.zarr_io import DenseZarrIO, CelltypeDenseZarrIO
import numpy as np


random.seed(0)

# Configuration for Gencode
gencode_config = {
    "assembly": "hg38",
    "version": 40,
    "gtf_dir": "/pmglocal/alb2281/repos/caesar/data"
}
# Configuration for the dataset
dataset_config = {
    "zarr_dirs": ["/pmglocal/alb2281/get/get_data/htan_gbm_dense.zarr"],
    "genome_seq_zarr": {'hg38': "/pmglocal/alb2281/get/get_data/hg38.zarr"},
    "genome_motif_zarr": "/pmglocal/alb2281/get/get_data/hg38_motif_result.zarr",
    "insulation_paths": [
        "/pmglocal/alb2281/repos/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather",
        "/pmglocal/alb2281/repos/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather"
    ],
    "peak_name": "fetal_gbm_peaks_open_exp",
    "leave_out_chromosomes": None,
    "use_insulation": True,
    "additional_peak_columns": ["Expression_positive", "Expression_negative", "aTPM", "Count", "TSS"],
    "n_peaks_upper_bound": 900,
    "keep_celltypes": "Tumor.htan_gbm.C3N-03186_CPT0206880004_snATAC_GBM_Tumor.2048",
    "center_expand_target": 0,
    "random_shift_peak": 0,
    "peak_count_filter": 0,
    'mask_ratio': 0,
    "padding": 0,
    "hic_path": None
}

# %%
hg38 = DenseZarrIO('/pmglocal/alb2281/get/get_data/hg38.zarr')
gencode = Gencode(**gencode_config)

# %%
cdz = CelltypeDenseZarrIO("/pmglocal/alb2281/get/get_data/htan_gbm_dense.zarr")

gene_list = ["TERT"]

# %%
dataset = InferenceDataset(
    assembly='hg38', gencode_obj={'hg38': gencode}, **dataset_config, gene_list=gene_list)
# %%
cfg = ReferenceRegionMotifConfig(
    data='fetal_gbm_peak_motif_v1.hg38.zarr',
    motif_scaler=1.3)


# %%
rrm = ReferenceRegionMotif(cfg)
# %%
rrd = InferenceReferenceRegionDataset(
    rrm, dataset, quantitative_atac=True, sampling_step=450)


# %%
config = OmegaConf.load(
    '/pmglocal/alb2281/repos/get_model/get_model/config/model/GETRegionFinetune.yaml')
model = hydra.utils.instantiate(config)['model']
#%%
import hydra
from get_model.config.config import Config
import torch.utils.data
from hydra.core.global_hydra import GlobalHydra

def load_config(config_name):
    # Initialize Hydra to load the configuration
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../../get_model/get_model/config", version_base="1.3")
    cfg = hydra.compose(config_name=config_name)
    return cfg

checkpoint_cfg= load_config('eval_gbm_fetal_ref_region')
# %%

checkpoint = torch.load("/pmglocal/alb2281/get/get_ckpts/watac_checkpoint_best.pth")
checkpoint = rename_state_dict(checkpoint['model'], checkpoint_cfg.finetune.rename_config)

model.load_state_dict(checkpoint)

dl = torch.utils.data.DataLoader(
    rrd, batch_size=1, shuffle=False, num_workers=4)



def recursive_to_device(data, device='cuda', dtype=torch.bfloat16):
    if isinstance(data, dict):
        return {k: recursive_to_device(v, device, dtype) for k, v in data.items()}
    elif isinstance(data, list):
        return [recursive_to_device(v, device, dtype) for v in data]
    elif isinstance(data, torch.Tensor):
        return data.to(device, dtype=dtype)
    else:
        return data


def interpret_step(model, batch, batch_idx, layer_names: List[str] = None, focus: int = None):
    target_tensors = {}
    hooks = []
    batch = recursive_to_device(batch)
    input = model.get_input(batch)
    input = recursive_to_device(input)
    assert focus is not None, "Please provide a focus position for interpretation"
    assert layer_names is not None, "Please provide a list of layer names for interpretation"
    for layer_input_name in layer_names:
        assert layer_input_name in model.get_layer_names(
        ), f"{layer_input_name} is not valid, valid layers are: {model.get_layer_names()}"

    # Register hooks to capture the target tensors
    def capture_target_tensor(name):
        def hook(module, input, output):
            # Retain the gradient of the target tensor
            output.retain_grad()
            target_tensors[name] = output
        return hook

    if layer_names is None or len(layer_names) == 0:
        target_tensors['input'] = input
        for key, tensor in input.items():
            tensor.requires_grad = True
    else:
        target_tensors['input'] = input
        for key, tensor in input.items():
            tensor.requires_grad = True
        for layer_name in layer_names:
            layer = model.get_layer(layer_name)
            hook = layer.register_forward_hook(
                capture_target_tensor(layer_name))
            hooks.append(hook)

    # Forward pass
    output = model(**input)
    pred, obs = model.before_loss(output, batch)
    # Remove the hooks after the forward pass
    # for hook in hooks:
    #     hook.remove()
    # Compute the jacobian of the output with respect to the target tensor
    jacobians = {}
    for target_name, target in obs.items():
        if target_name != 'exp':
            continue
        jacobians[target_name] = {}
        for i in range(target.shape[-1]):
            output = model(**input)
            pred, obs = model.before_loss(output, batch)
            jacobians[target_name][str(i)] = {}
            mask = torch.zeros_like(target).to('cuda')
            mask[:, focus, i] = 1
            pred[target_name].backward(mask)
            for layer_name, layer in target_tensors.items():
                if isinstance(layer, torch.Tensor):
                    if layer.grad is None:
                        continue
                    jacobians[target_name][str(
                        i)][layer_name] = layer.grad.detach().cpu().float().numpy()
                    layer.grad.zero_()
                elif isinstance(layer, dict):
                    for layer_input_name, layer_input in layer.items():
                        if layer_input.grad is None:
                            continue
                        jacobians[target_name][str(
                            i)][layer_name] = layer_input.grad.detach().cpu().float().numpy()
                        layer_input.grad.zero_()
    pred = recursive_numpy(recursive_detach(pred))
    obs = recursive_numpy(recursive_detach(obs))
    jacobians = recursive_numpy(jacobians)
    target_tensors = recursive_numpy(target_tensors)
    return pred, obs, jacobians, target_tensors


# %%
with torch.autocast(dtype=torch.bfloat16, device_type='cuda'):
    model.to('cuda')
    for i, data in tqdm(enumerate(dl)):
        if i>0:
            break
        input_data = model.get_input(data)
        result = model(input_data['region_motif'].cuda())
        pred, obs, jacobians, target_tensors = interpret_step(
            model, data, 0, ['region_embed'], data['tss_peak'])
        for i in range(len(data['gene_name'])):
            peaks = pd.DataFrame(
                data['peak_coord'][i].squeeze().numpy(), columns=['Start', 'End'])
            peaks['Chromosome'] = data['chromosome'][i]
            peaks['gene_name'] = data['gene_name'][i]

            jacobian_norm = np.linalg.norm(jacobians['exp'][str(
                data['strand'].item())]['region_embed'][i], axis=1)
            jacobian = jacobians['exp'][str(data['strand'].item())]['region_embed'][i]
            input_jacobian = jacobian = jacobians['exp'][str(data['strand'].item())]['input'][i]
            input_matrix = target_tensors['input']['region_motif'][i].squeeze()

            all_tss_peak = data['all_tss_peak'][i].squeeze().numpy()
            all_tss_peak = all_tss_peak[all_tss_peak > 0]
            atac = data['region_motif'][i].squeeze().detach().cpu().numpy()[
                :, -1]
            peaks['atac'] = atac
            peaks['jacobian_norm'] = jacobian_norm
            
            results_dir = "/pmglocal/alb2281/repos/get_model/analysis/results/unnorm_input_zeroshot_TERT_get"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            peaks.to_csv(f"{results_dir}/unnorm_input_zeroshot_{data['gene_name'][i]}_get.csv")
            np.save(f"{results_dir}/jacobian_{data['gene_name'][i]}.npy", jacobian)
            np.save(f"{results_dir}/input_jacobian_{data['gene_name'][i]}.npy", input_jacobian)
            np.save(f"{results_dir}/input_matrix_{data['gene_name'][i]}.npy", input_matrix)
