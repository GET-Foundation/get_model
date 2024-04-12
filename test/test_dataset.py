# %%
from tqdm import tqdm
import torch
import argparse
from dataset.dataset import build_dataset
import sys
sys.path.append("..")
# use k562_cut0.04 as an example.
# simulate an arugment parser
args = argparse.Namespace(
    data_set='Expression',
    data_type='k562',
    leave_out_chromosomes="",
    leave_out_celltypes="k562_bulk_cut0.03,k562_bulk_cut0.04,k562_bulk_cut0.05,k562_bulk_cut0.07,k562_cut0.03,k562_cut0.04,k562_cut0.05",
    quantitative_atac=False,
    mask_tss=True,
    sampling_step=100,
    target_sequence_length=200,
    shift=100,
    num_region_per_sample=200,
    use_seq=False,
    mask_ratio=0.1,
    data_path='/pmglocal/xf2217/pretrain_human_bingren_shendure_apr2023',

)
# %%
pretrain_dataset = build_dataset(is_train=True, args=args)
# %%
sampler_train = torch.utils.data.DistributedSampler(
    pretrain_dataset, num_replicas=1, rank=0, shuffle=True)

data_loader_train = torch.utils.data.DataLoader(
    pretrain_dataset, sampler=sampler_train,
    batch_size=8,
    num_workers=32,
    pin_memory=False,
    drop_last=True,
)

# %%
for i, batch in tqdm(enumerate(data_loader_train)):
    sample, mask, ctcf = batch


# %%
# save the dataset
torch.save(pretrain_dataset, 'pretrain_dataset.pth')
# %%
