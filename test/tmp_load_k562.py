# %%
import logging

from get_model.dataset.collate import get_rev_collate_fn
from get_model.dataset.zarr_dataset import PretrainDataset, ZarrDataPool, PreloadDataPack, CelltypeDenseZarrIO

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
#%% 
pretrain = PretrainDataset(zarr_dirs=['/pmglocal/alb2281/get/get_data/encode_hg38atac_dense.zarr',
                            ],
                           genome_seq_zarr='/pmglocal/alb2281/get/get_data/hg38.zarr', 
                           genome_motif_zarr='/pmglocal/alb2281/get/get_data/hg38_motif_result.zarr', insulation_paths=[
                           '/pmglocal/alb2281/repos/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather', '/pmglocal/alb2281/repos/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather'], peak_name='peaks_q0.05_fetal_joint_tissue_open_exp', preload_count=100, n_packs=1,
                           max_peak_length=5000, center_expand_target=500, n_peaks_lower_bound=10, n_peaks_upper_bound=100, use_insulation=False, is_train=False, dataset_size=65536, additional_peak_columns=None, hic_path='/burg/pmg/users/xf2217/get_data/4DNFI2TK7L2F.hic')
pretrain.__len__()
# %%

pretrain.datapool.peaks_dict
# %%
len(pretrain.datapool.peaks_dict)
# %%
pretrain.datapool.peaks_dict.keys()
# %%
peaks_dict = pretrain.datapool.peaks_dict["k562.encode_hg38atac.ENCFF128WZG.max"]
# %%

peaks_dict.to_csv("/pmglocal/alb2281/repos/get_model/analysis/k562_orig_peaks_dict.csv")

# %%
peaks_dict
# %%

peaks_dict["name"] = peaks_dict["Chromosome"].astype(str) + ":" + peaks_dict["Start"].astype(str) + "-" + peaks_dict["End"].astype(str)
# %%

peaks_dict.to_csv("/pmglocal/alb2281/repos/get_model/analysis/k562_cage_data/k562_orig_peaks_dict.csv")
