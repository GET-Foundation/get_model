import pytest
import os
import hydra
from get_model.config.config import Config
from hydra.core.global_hydra import GlobalHydra
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any
# disable DeprecationWarning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
def load_config(config_name: str):
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../get_model/config", version_base="1.3")
    return hydra.compose(config_name=config_name)

def get_dataloader(cfg: Any, stage: str):
    from get_model.run_everything import EverythingDataModule
    from get_model.run_ref_region import ReferenceRegionDataModule
    from get_model.run_region import RegionDataModule
    from get_model.run import GETDataModule

    if cfg.type == 'everything':
        dm = EverythingDataModule(cfg)
    elif cfg.type == 'reference_region':
        dm = ReferenceRegionDataModule(cfg)
    elif cfg.type == 'region':
        dm = RegionDataModule(cfg)
    elif cfg.type == 'nucleotide':
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

def visualize_sample(batch: Dict[str, Any], output_dir: str, stage: str):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    batch_size = len(batch['celltype_id'])
    split_track = torch.split(batch['sample_track'].flatten(), batch['chunk_size'])
    split_track = np.array([i.sum() for i in split_track]).reshape(batch_size, -1)[:, :-1]
    
    sns.scatterplot(x=batch['atpm'].numpy().flatten(), 
                    y=np.log10(split_track.flatten()/batch['metadata'][0]['libsize']*1e6+1), 
                    ax=ax)
    
    ax.set_xlabel('ATPM')
    ax.set_ylabel('Log10 Normalized Track Sum')
    ax.set_title(f'{stage.capitalize()} - ATPM vs Normalized Track Sum')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{stage}_atpm_vs_track.png'))
    plt.close()

    return np.corrcoef(batch['atpm'].numpy().flatten(), 
                       np.log10(split_track.flatten()/batch['metadata'][0]['libsize']*1e6+1))[0,1]

def visualize_motifs(batch: Dict[str, Any], output_dir: str, stage: str):
    motifs = batch['region_motif'][0]  # Assuming the first sample in the batch
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    sns.heatmap(motifs.T, cmap='viridis', ax=ax)
    ax.set_title(f'{stage.capitalize()} - Motif Heatmap')
    ax.set_xlabel('Genomic Position')
    ax.set_ylabel('Motif')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{stage}_motif_heatmap.png'))
    plt.close()

@pytest.fixture(scope="module")
def config(request):
    return load_config(request.param)

@pytest.fixture(scope="module")
def output_dir(request):
    dir_name = f'test_results_{request.param}'
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

@pytest.mark.parametrize("config,output_dir", [
    ("eval_k562_fetal_ref_region_k562_hic_oe", "eval_k562_fetal_ref_region_k562_hic_oe"),
    # Add more configurations as needed
], indirect=True)
class TestDataset:

    @pytest.mark.parametrize("stage", ["fit", "validate", "predict"])
    def test_dataloader(self, config, output_dir, stage):
        dataloader, dm = get_dataloader(config, stage)
        batch = next(iter(dataloader))

        # Test batch structure
        assert isinstance(batch, dict), f"{stage} batch should be a dictionary"
        
        # Test data types and shapes
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                assert value.dim() > 0, f"{key} in {stage} batch should not be a 0-dim tensor"
                print(f"{stage} - {key}: shape {value.shape}")
            else:
                print(f"{stage} - {key}: {type(value)}")

        # Visualizations and specific tests
        if 'atpm' in batch and 'sample_track' in batch:
            corr = visualize_sample(batch, output_dir, stage)
            assert corr > 0.4, f"Correlation in {stage} should be > 0.4, but got {corr}"

        if 'region_motif' in batch:
            visualize_motifs(batch, output_dir, stage)
            assert batch['region_motif'].dim() == 3, f"region_motif in {stage} should be 3D, but got {batch['region_motif'].dim()}D"

        # Test for 'everything' type datasets
        if 'rrd' in batch and 'zarr' in batch:
            assert set(batch['rrd'].keys()) == set(batch['zarr'].keys()), f"Keys in 'rrd' and 'zarr' should match for {stage}"

        # Add more specific tests as needed

if __name__ == "__main__":
    pytest.main([__file__, "-v"])