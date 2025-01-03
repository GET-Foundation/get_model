import glob
import os

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import zarr
from omegaconf import DictConfig

from get_model.config.config import *
from get_model.dataset.zarr_dataset import MPRADataset
from get_model.model.model import *
from get_model.model.modules import *
from get_model.run import LitModel
from get_model.utils import setup_trainer


class MPRADataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        if stage == 'predict' or stage is None:
            self.dataset_predict = MPRADataset(
                root=self.cfg.data.root,
                metadata_path=self.cfg.data.metadata_path,
                num_region_per_sample=self.cfg.model.num_region_per_sample,
                mpra_feather_path=self.cfg.mpra.input_feather,
                focus=self.cfg.mpra.focus,
                data_type=self.cfg.data.type,
                quantitative_atac=self.cfg.data.quantitative_atac
            )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_predict,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.machine.num_workers,
            shuffle=False
        )

class MPRALitModel(LitModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def predict_step(self, batch, batch_idx):
        _, preds, _ = self._shared_step(batch, batch_idx, stage='predict')
        return preds

def save_mpra_data(predictions, cfg: DictConfig, dm: MPRADataModule):
    """
    Save MPRA predictions and data.
    
    Args:
    predictions (list): List of prediction dictionaries.
    cfg (DictConfig): Configuration object.
    dm (MPRADataModule): Data module containing the dataset.
    """
    # Process predictions
    all_preds = np.concatenate([p['exp'] for p in predictions])
    
    # Reshape predictions to match the original structure
    reshaped_preds = all_preds.reshape(-1, cfg.model.num_region_per_sample, 2)
    
    # Create the output directory structure
    base_output_dir = os.path.join(cfg.mpra.output_dir, cfg.run.project_name)
    os.makedirs(base_output_dir, exist_ok=True)

    # Save predictions for each chunk
    chunk_size = cfg.mpra.chunk_size
    for i, chunk_start in enumerate(range(0, len(reshaped_preds), chunk_size)):
        chunk_end = chunk_start + chunk_size
        chunk_preds = reshaped_preds[chunk_start:chunk_end]
        
        # Create chunk directory
        chunk_dir = os.path.join(base_output_dir, f'chunk_{i}', 'rna', cfg.data.celltype)
        os.makedirs(chunk_dir, exist_ok=True)
        
        # Save as Zarr
        zarr_path = os.path.join(chunk_dir, 'predictions.zarr')
        z = zarr.open(zarr_path, mode='w')
        z.create_dataset('predictions', data=chunk_preds, chunks=(1000, chunk_preds.shape[1], 2))

    # Save MPRA data with predictions
    mpra_data = dm.dataset_predict.mpra
    for i, chunk_start in enumerate(range(0, len(mpra_data), chunk_size)):
        chunk_end = chunk_start + chunk_size
        chunk = mpra_data.iloc[chunk_start:chunk_end].copy()
        chunk['prediction'] = reshaped_preds[chunk_start:chunk_end, cfg.mpra.focus, 0]
        chunk.to_feather(os.path.join(base_output_dir, f'mpra_with_predictions_chunk{i}.feather'))

    logging.debug(f"Predictions saved in {base_output_dir}")

def load_get_data(path, data_type='peak', promoter_only=False, repeats=600):
    """
    Load data from Zarr format.
    
    Args:
    path (str): Path to the directory containing Zarr files.
    data_type (str): Type of data ('peak' or 'nonpeak').
    promoter_only (bool): Whether to filter for promoters only.
    repeats (int): Number of repeats in the data.
    
    Returns:
    pandas.DataFrame: Processed data.
    """
    pred_all = []
    for chunk_dir in sorted(glob.glob(os.path.join(path, 'chunk_*'))):
        zarr_path = os.path.join(chunk_dir, 'rna', '*', 'predictions.zarr')
        zarr_files = glob.glob(zarr_path)
        if zarr_files:
            z = zarr.open(zarr_files[0], mode='r')
            predictions = z['predictions'][:]
            pred_all.append(predictions[:, 100, :].max(1).reshape(-1, repeats))
    
    pred_all = np.vstack(pred_all)
    pred_all_mean = pred_all.mean(axis=1)
    pred_all_std = pred_all.std(axis=1)
    
    # Load and concatenate all chunked feather files
    peak_chunks = []
    for chunk_file in sorted(glob.glob(os.path.join(path, f'mpra_with_predictions_chunk*.feather'))):
        peak_chunks.append(pd.read_feather(chunk_file))
    peak = pd.concat(peak_chunks, ignore_index=True)
    
    peak['pred'] = pred_all_mean
    peak['pred_std'] = pred_all_std
    
    if promoter_only:
        peak = peak.query('Name.str.contains("ENSG")')
    return peak

@hydra.main(config_path="config", config_name="mpra_config")
def main(cfg: DictConfig):
    model = MPRALitModel(cfg)
    dm = MPRADataModule(cfg)
    
    trainer, _ = setup_trainer(cfg)
    
    predictions = trainer.predict(model, datamodule=dm)
    
    # Save predictions and MPRA data
    save_mpra_data(predictions, cfg, dm)

if __name__ == "__main__":
    main()