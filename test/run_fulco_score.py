# %%
from tqdm import tqdm
import hicstraw
from minlora.model import add_lora_by_name
from get_model.utils import rename_lit_state_dict
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
from caesar.io.zarr_io import DenseZarrIO
import numpy as np
np.bool = np.bool_

random.seed(0)

# %%
# Configuration for Gencode
gencode_config = {
    "assembly": "hg38",
    "version": 40,
    "gtf_dir": "/home/xf2217/Projects/caesar/data/"
}
# Configuration for the dataset
dataset_config = {
    "zarr_dirs": ["/home/xf2217/Projects/get_data/encode_hg38atac_dense.zarr"],
    "genome_seq_zarr": {'hg38': "/home/xf2217/Projects/get_data/hg38.zarr"},
    "genome_motif_zarr": "/home/xf2217/Projects/get_data/hg38_motif_result.zarr",
    "insulation_paths": [
        "/home/xf2217/Projects/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather",
        "/home/xf2217/Projects/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather"
    ],
    "peak_name": "peaks_q0.05_fetal_joint_tissue_open_exp",
    "leave_out_chromosomes": None,
    "use_insulation": True,
    "additional_peak_columns": ["Expression_positive", "Expression_negative", "aTPM", "TSS"],
    "n_peaks_upper_bound": 900,
    "keep_celltypes": "k562.encode_hg38atac.ENCFF257HEE.max",
    "center_expand_target": 0,
    "random_shift_peak": 0,
    "peak_count_filter": 0,
    'mask_ratio': 0,
    "padding": 0,
    "hic_path": None
}
# %%
hg38 = DenseZarrIO('/home/xf2217/Projects/get_data/hg38.zarr')
gencode = Gencode(**gencode_config)
# %%
gene_list = np.loadtxt(
    '../genes.txt', dtype=str)

dataset = InferenceDataset(
    assembly='hg38', gencode_obj={'hg38': gencode}, **dataset_config, gene_list=gene_list)
# %%
cfg = ReferenceRegionMotifConfig(
    data='k562.ENCFF257HEE.encode_hg38atac.peak_motif.zarr',
    motif_scaler=1.3)
rrm = ReferenceRegionMotif(cfg)
# %%
rrd = InferenceReferenceRegionDataset(
    rrm, dataset, quantitative_atac=True, sampling_step=450)


# %%
# load yaml config from /home/xf2217/Projects/get_model/get_model/config/model/GETRegionFinetuneExpHiCABC.yaml
config = OmegaConf.load(
    '/home/xf2217/Projects/get_model/get_model/config/model/GETRegionFinetuneExpHiCABC.yaml')
model = hydra.utils.instantiate(config)['model']
# load GETRegionFinetune_k562_abc/kb3g6ciz/checkpoints/best.ckpt
# checkpoint = torch.load('/home/xf2217/Projects/get_model/GETRegionFinetune_k562_abc/idhidt4u/checkpoints/best.ckpt')
# checkpoint = torch.load('/home/xf2217/Projects/get_model/GETRegionFinetune_k562_abc/ah9um258/checkpoints/best.ckpt')
# checkpoint = torch.load('/home/xf2217/Projects/get_model/GETRegionFinetune_k562_abc/5rn0l6zi/checkpoints/best.ckpt')
checkpoint = torch.load(
    '/home/xf2217/Projects/get_model/GETRegionFinetune_k562_abc/wh4s1c1p/checkpoints/best.ckpt')
lora_config = {  # specify which layers to add lora to, by default only add to linear layers
    nn.Linear: {
        "weight": partial(LoRAParametrization.from_linear, rank=8),
    },
}

add_lora_by_name(model, ['head_exp', 'region_embed', 'encoder'], lora_config)

model.load_state_dict(rename_lit_state_dict(checkpoint['state_dict']))
# %%
# side by side heatmap
hic = hicstraw.HiCFile(
    "/home/xf2217/Projects/encode_hg38atac/raw/ENCFF621AIY.hic")


def get_hic_from_idx_for_tss(hic, csv, start, end, resolution=5000, method='observed'):
    csv_region = csv.iloc[start:end]
    chrom = csv_region.iloc[0].Chromosome
    start = csv_region.iloc[0].Start // resolution
    end = csv_region.iloc[-1].End // resolution + 1
    # if (end-start) * resolution > 4000000:
    #     return None

    hic_idx = np.array([row.Start // resolution - start +
                       1 for _, row in csv_region.iterrows()])
    mzd = hic.getMatrixZoomData(
        chrom, chrom, method, "SCALE", "BP", resolution)
    numpy_matrix = mzd.getRecordsAsMatrix(
        start * resolution, end * resolution, start * resolution, end * resolution)
    if numpy_matrix.shape[0] < len(hic_idx):
        dst = np.zeros((len(hic_idx), len(hic_idx)))

    dst = np.log10(numpy_matrix[hic_idx, :][:, hic_idx]+1)
    return dst


# %%
dl = torch.utils.data.DataLoader(
    rrd, batch_size=1, shuffle=False, num_workers=4)
# %%


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
        result = model(input_data['region_motif'].cuda(),
                       input_data['distance_map'].cuda())
        pred, obs, jacobians, target_tensors = interpret_step(
            model, data, 0, ['region_embed'], data['tss_peak'])
        predicted_hic = model.distance_contact_map(
            input_data['distance_map'].cuda())
        for i in range(len(data['gene_name'])):
            peaks = pd.DataFrame(
                data['peak_coord'][i].squeeze().numpy(), columns=['Start', 'End'])
            peaks['Chromosome'] = data['chromosome'][i]
            peaks['gene_name'] = data['gene_name'][i]

            jacobian_norm = np.linalg.norm(jacobians['exp'][str(
                data['strand'].item())]['region_embed'][i], axis=1)
            # arr = get_hic_from_idx_for_tss(
            #     hic, peaks, 0, len(peaks), 5000, 'observed')

            all_tss_peak = data['all_tss_peak'][i].squeeze().numpy()
            all_tss_peak = all_tss_peak[all_tss_peak > 0]
            atac = data['region_motif'][i].squeeze().detach().cpu().numpy()[
                :, -1]
            abc = pred['abc'][0][all_tss_peak]
            predicted_hic = predicted_hic[i][0][all_tss_peak].mean(
                0).detach().cpu().numpy()
            # obs_hic = arr[all_tss_peak].mean(0)
            # if has 2 dim
            if len(abc.shape) == 2:
                abc = abc.max(0)
            peaks['abc'] = abc
            peaks['atac'] = atac
            peaks['jacobian_norm'] = jacobian_norm
            peaks['predicted_hic'] = predicted_hic
            # peaks['obs_hic'] = obs_hic
            peaks.to_csv(f"dnase_jacob_abc_diff_hic/{data['gene_name'][i]}_get.csv")


# %%

# fig, ax = plt.subplots(1, 3, figsize=(12, 4))
# sns.heatmap(jacobians['exp'][str(data['strand'].item())]['region_embed'][0], ax=ax[0], cmap='coolwarm',vmin=-0.1, vmax=0.1, cbar=False, xticklabels=False, yticklabels=False)
# sns.heatmap(target_tensors['region_embed'][0]*jacobians['exp'][str(data['strand'].item())]['region_embed'][0], ax=ax[1], cmap='coolwarm',  vmin=-0.1, vmax=0.1, cbar=False, xticklabels=False, yticklabels=False)
# sns.heatmap(arr, ax=ax[2], cmap='coolwarm', cbar=False, xticklabels=False, yticklabels=False)
# plt.tight_layout()

# # %%
