# %%
preds_dir = "/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/preds"

# %%
import os
import numpy as np

# %%
exp_gt = np.load("/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/data/leaveout_celltypes/cerebellum_1.exp.npy")
# %%
# average of two columns in ground truth
avg_gt = np.mean(exp_gt, axis=1)
# %%

# iterate over prediction files and calculate correlation with ground truth
cage_preds = []
for pred_file in os.listdir(preds_dir):
    # if is a file
    if os.path.isfile(os.path.join(preds_dir, pred_file)):
        pred_batch = np.load(os.path.join(preds_dir, pred_file))
        for item in pred_batch:
            cage_preds.append([int(item[0]), item[1]])
# %%

tss = np.load("/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/data/leaveout_celltypes/cerebellum_1.tss.npy")
# %%
# find rows where either column is True
tss_rows = np.where(np.any(tss, axis=1))
# %%
