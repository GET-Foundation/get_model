#%%
import numpy as np
import zarr
from numcodecs import Blosc
from typing import Any, Tuple, List
import pandas as pd
from tqdm import tqdm
import sys 
from dataset.zarr_dataset import ZarrIO, DenseZarrIO

# %%
dz = DenseZarrIO('/manitou/pmg/users/xf2217/atac_rna_data_processing/test/hg38.zarr')
# %%
region_list = [(i * 2000, (i + 1) * 2000) for i in range(200)]

for i in tqdm(range(10)):
    a = dz.get_regions('chr1', region_list)

# %%
