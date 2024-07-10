#%%
import hydra
from get_model.config.config import Config
import torch.utils.data
from hydra.core.global_hydra import GlobalHydra
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_config(config_name):
    # Initialize Hydra to load the configuration
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../get_model/config", version_base="1.3")
    cfg = hydra.compose(config_name=config_name)
    return cfg

def get_dataloader(cfg: Config, stage='fit', type='everything'):
    # Depending on the stage, create the appropriate DataModule and DataLoader
    from get_model.run_everything import EverythingDataModule
    from get_model.run_ref_region import ReferenceRegionDataModule
    from get_model.run import GETDataModule
    if type == 'everything':
        dm = EverythingDataModule(cfg)
    elif type == 'ref_region':
        dm = ReferenceRegionDataModule(cfg)
    else:
        dm = GETDataModule(cfg)
    dm.setup(stage)

    if stage == 'fit':
        return dm.train_dataloader(), dm
    elif stage == 'validate':
        return dm.val_dataloader(), dm
    elif stage == 'test':
        return dm.test_dataloader(), dm
    elif stage == 'predict':
        return dm.predict_dataloader(), dm
    else:
        raise ValueError(f"Invalid stage: {stage}")

def inspect_dataloader(config_name, stage='fit', type='everything', num_batches=1):
    cfg = load_config(config_name)
    dataloader, dm = get_dataloader(cfg, stage, type)

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        print(f"Batch {i + 1}:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape {value.shape}")
            else:
                print(f"{key}: {value}")
        print("\n")
    return cfg, dm

#%%
# Specify the configuration name you want to use
config_name = "eval_k562_fetal_ref_region"

# Inspect the dataloader for the 'fit' stage
cfg,  dm = inspect_dataloader(config_name, stage='predict', type='ref_region', num_batches=2)

# %%
# assert correlation of exp with atpm > 0.4
for celltype_id, peak in dm.dataset_val.zarr_dataset.datapool.peaks_dict.items():
    peak_df = peak.copy()
    peak_df['exp'] = peak_df['Expression_positive'] + peak_df['Expression_negative']
    if peak_df['exp'].corr(peak_df['aTPM']) < 0.4:
        print(f"Celltype {celltype_id} has a low correlation of {peak_df['exp'].corr(peak_df['aTPM'])}")
# %%
# assert the boundary of the peak is within the metadata start and end
for i in range(len(batch['peak_coord'])):
    assert batch['peak_coord'][i].min() > batch['metadata'][i]['start'] & batch['peak_coord'][i].max() < batch['metadata'][i]['end'] 


# %%
assert sum(sum(batch['peak_coord'][0] - (batch['celltype_peaks'][0]+batch['metadata'][0]['start']+batch['metadata'][0]['i_start']))) == 0
# %%
def visualize_sample(batch, idx=1):
    fig, ax = plt.subplots(1,1, figsize=(5, 5))
    # split batch['sample_track'][idx] based on batch['chunk_size'] in to B*R segment. len(batch['chunk_size']) = B*R
    batch_size = len(batch['celltype_id'])
    split_track = torch.split(batch['sample_track'].flatten(), batch['chunk_size'])
    split_track = np.array([i.sum() for i in split_track]).reshape(batch_size, -1)[:, :-1]
    sns.scatterplot(x=batch['atpm'].numpy().flatten(), y=np.log10(split_track.flatten()/batch['metadata'][0]['libsize']*1e6+1), ax=ax)
    # assert spearman correlatation > 0.9
    assert np.corrcoef(batch['atpm'].numpy().flatten(), np.log10(split_track.flatten()/batch['metadata'][0]['libsize']*1e6+1))[0,1] > 0.9
    return split_track
# %%
split_track = visualize_sample(batch, 1)
# %%
data_dict = dm.dataset_val.data_dict
# %%
batch['chromosome'][0]
# %%
rrd_item = dm.dataset_val.get_item_from_coord(batch['chromosome'][0], batch['peak_coord'][0][0][0].item(), batch['peak_coord'][0][-1][1].item(), batch['celltype_id'][0])
# %%
