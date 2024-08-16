#%%
import logging
import os

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import zarr
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig
from scipy.sparse import coo_matrix, load_npz, save_npz
from get_model.config.config import *
from get_model.dataset.zarr_dataset import (RegionDataset)
from get_model.model.model_refactored import *
from get_model.model.modules import *
from get_model.run import LitModel

class MPRADataset(RegionDataset):
    def __init__(self, root, metadata_path, num_region_per_sample, mpra_feather_path, focus, data_type="fetal", quantitative_atac=False):
        super().__init__(root, metadata_path, num_region_per_sample, data_type=data_type, quantitative_atac=quantitative_atac)
        self.mpra_feather_path = mpra_feather_path
        self.focus = focus
        self.mpra = pd.read_feather(self.mpra_feather_path)
        self.load_mpra_data()

    def load_mpra_data(self):
        # Load the original peak data
        cell_data = load_npz(f"{self.root}/{self.data_type}.{'natac' if self.quantitative_atac else 'watac'}.npz")
        annot = pd.read_csv(f"{self.root}/{self.data_type}.csv", index_col=0)

        # Generate sample list
        self.sample_list = []
        for chr in annot.Chromosome.unique():
            idx_sample_list = annot.index[annot['Chromosome'] == chr].tolist()
            idx_sample_start = idx_sample_list[0]
            idx_sample_end = idx_sample_list[-1]
            for i in range(idx_sample_start, idx_sample_end, 5):
                start_index = i
                end_index = i + self.num_region_per_sample
                self.sample_list.append((start_index, end_index))

        # Pre-sample indices for each MPRA entry
        self.sampled_indices = np.random.choice(range(len(self.sample_list)), size=len(self.mpra), replace=True)

        self.cell_data = cell_data
        self.annot = annot

    def __len__(self):
        return len(self.mpra)

    def __getitem__(self, idx):
        mpra_row = self.mpra.iloc[idx]
        sample_idx = self.sampled_indices[idx]
        start_index, end_index = self.sample_list[sample_idx]

        # Get the original peak data for the sampled region
        c_data = self.cell_data[start_index:end_index].toarray()

        # Insert MPRA data at the focus index
        c_data[self.focus] = mpra_row.values[1:] + c_data[self.focus]
        c_data[c_data > 1] = 1
        c_data[self.focus, 282] = 1

        # Create target data (all zeros for MPRA prediction)
        t_data = np.zeros((self.num_region_per_sample, 2))

        return {
            'region_motif': c_data.astype(np.float32),
            'mask': np.ones(self.num_region_per_sample),
            'exp_label': t_data.astype(np.float32)
        }
    

class MPRALitModel(LitModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def predict_step(self, batch, batch_idx):
        _, preds, _ = self._shared_step(batch, batch_idx, stage='predict')
        return preds

@hydra.main(config_path="config", config_name="mpra_config")
def main(cfg: DictConfig):
    # Create MPRADataset
    dataset = MPRADataset(
        root=cfg.data.root,
        metadata_path=cfg.data.metadata_path,
        num_region_per_sample=cfg.model.num_region_per_sample,
        mpra_feather_path=cfg.mpra.input_feather,
        focus=cfg.mpra.focus,
        data_type=cfg.data.type,
        quantitative_atac=cfg.data.quantitative_atac
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.machine.num_workers,
        shuffle=False
    )

    # Initialize model
    model = MPRALitModel(cfg)

    # Initialize trainer
    trainer = L.Trainer(
        accelerator=cfg.machine.accelerator,
        devices=cfg.machine.num_devices,
        logger=CSVLogger(save_dir=cfg.machine.output_dir, name=cfg.wandb.run_name),
        default_root_dir=cfg.machine.output_dir,
    )

    # Predict
    predictions = trainer.predict(model, dataloader)

    # Process and save predictions
    all_preds = np.concatenate([p['exp'] for p in predictions])
    
    # Save as NPZ
    save_npz(os.path.join(cfg.mpra.output_dir, 'preds.npz'), coo_matrix(all_preds))

    # Save as Zarr if specified
    if cfg.mpra.save_zarr:
        zarr_path = os.path.join(cfg.mpra.output_dir, 'predictions.zarr')
        z = zarr.open(zarr_path, mode='w')
        z.create_dataset('predictions', data=all_preds, chunks=(1000, all_preds.shape[1]))

    # Save MPRA data with predictions
    mpra_with_preds = dataset.mpra.copy()
    mpra_with_preds['prediction'] = all_preds[:, dataset.focus]  # Assuming predictions are for the focus position
    mpra_with_preds.to_feather(os.path.join(cfg.mpra.output_dir, 'mpra_with_predictions.feather'))

if __name__ == "__main__":
    main()