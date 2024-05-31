# %%
import numpy as np
import pandas as pd
tss_array = np.load("/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/data/leaveout_celltypes/cerebellum_1.tss.npy")

# %%
# filter to rows where first column is True
tss_array_plus = tss_array[tss_array[:,0], :]
# %%
# filter to rows where second column is True
tss_array_minus = tss_array[tss_array[:,1], :]
# %%
# filter to rows where both first and second column are True
tss_array_both = tss_array[tss_array[:,0] & tss_array[:,1], :]
# %%
# filter to rows where either first or second column is True
tss_array_either = tss_array[tss_array[:,0] | tss_array[:,1], :]
# %%
len(tss_array_plus), len(tss_array_minus), len(tss_array_both), len(tss_array_either)
# %%
len(tss_array_plus) + len(tss_array_minus) - len(tss_array_both) == len(tss_array_either)
# %%
# get indices for rows where either column is True
either_indices = np.where(tss_array[:,0] | tss_array[:,1])
# %%
astrocyte_file = "/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/data/leaveout_celltypes/cerebellum_1.csv"
astrocyte_df = pd.read_csv(astrocyte_file)
# %%
