# utils.py
#%%
import hydra
from get_model.config.config import Config
import torch.utils.data
from hydra.core.global_hydra import GlobalHydra

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
    return cfg, batch, dm

#%%
# Specify the configuration name you want to use
config_name = "eval_k562_fetal_nucleotide_motif_adaptor"

# Inspect the dataloader for the 'fit' stage
cfg, batch, dm = inspect_dataloader(config_name, stage='fit', type='everything', num_batches=50)

# %%
# assert correlation of exp with atpm > 0.4
for celltype_id, peak in dm.dataset_train.zarr_dataset.datapool.peaks_dict.items():
    peak_df = peak.copy()
    peak_df['exp'] = peak_df['Expression_positive'] + peak_df['Expression_negative']
    if peak_df['exp'].corr(peak_df['aTPM']) < 0.4:
        print(f"Celltype {celltype_id} has a low correlation of {peak_df['exp'].corr(peak_df['aTPM'])}")
# %%
# assert the boundary of the peak is within the metadata start and end
for i in range(len(batch['peak_coord'])):
    assert batch['peak_coord'][i].min() > batch['metadata'][i]['start'] & batch['peak_coord'][i].max() < batch['metadata'][i]['end'] 

