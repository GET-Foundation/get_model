# Configuration System

This document explains how to use the configuration system in our project, which is based on the Hydra framework. We'll cover the main components of the configuration system and how to override configurations using the command line.


## Using the Configuration System

To use the configuration system, you typically create a YAML file for your specific task or experiment. This file will inherit from the base configurations and override or add specific settings.

For example, in `fetal_region_erythroblast.yaml`:

```yaml
defaults:
    base_region_config
    model/GETRegionFinetune@here
    finetune/v1_finetune@finetune
    machine/pc_region_v1
    _self_
assembly: "hg19"
dataset:
    root: ${machine.data_path}
    metadata_path: 'cell_type_align.txt'
    # ... other dataset settings ...
training:
    save_ckpt_freq: 1
    epochs: 100
    # ... other training settings ...
# ... other configuration sections ...
```
This configuration file inherits from `base_region_config` and includes specific model, finetune, and machine configurations. It then overrides or adds specific settings for the fetal region erythroblast task.

## Overriding Configurations via Command Line

Hydra allows you to override configuration values directly from the command line. This is useful for quick experiments or adjustments without modifying the configuration files.

### Basic Syntax

The general syntax for overriding configurations is:
```bash
python script.py key1=value1 key2.nested_key=value2
```

### Example

To run a script with specific overrides, you might use a command like this:
```bash
python get_model/debug/debug_run_region.py \
--config-name fetal_region_erythroblast \
stage=predict \
machine.batch_size=2 \
wandb.run_name=eryth_test_compatibility
```
This command does the following:
1. Uses the `fetal_region_erythroblast` configuration
2. Sets the `stage` to "predict"
3. Changes the `batch_size` in the machine configuration to 2
4. Sets the Weights & Biases run name to "eryth_test_compatibility"

### Common Overrides

Some common configuration overrides include:

- Changing the dataset: `dataset.root=/path/to/new/data`
- Adjusting training parameters: `training.epochs=50 optimizer.lr=0.001`
- Modifying model architecture: `model.cfg.num_layers=8`
- Changing output directory: `hydra.run.dir=/path/to/output`

Remember that you can override any configuration value, including nested values, using dot notation.

## Components of the Configuration System

Our configuration system is defined with configuration template dataclass in `get_model/config/config.py` and include:
   - Main configuration classes, can be divided into four classes based on the model type used:
     - `Config`: The main configuration class for nucleotide-level model
     - `ReferenceRegionConfig`: The main configuration class  for reference region model
     - `EverythingConfig`: The main configuration class  for "everything" model
     - `RegionConfig`: The main configuration class  for region-specific model (The original model used in the paper)
   - `ModelConfig`: Model configuration (e.g. number of layers, number of heads, etc.)
   - `MachineConfig`: Machine configuration (e.g. number of GPUs, number of workers, etc.)
   - `TrainingConfig`: Training configuration (e.g. number of epochs, learning rate, etc.)
   - `FinetuneConfig`: Fine-tuning configuration (e.g. use LoRA or not, pretrained checkpoint, etc.)
   - `DatasetConfig`: Dataset configuration (e.g. leave out cell types, leave out chromosomes, etc.)
   - `RunConfig`: Run configuration (e.g. job name, project name, etc.)

### Main Configurations

Corresponding to the main configuration classes, these are YAML files in the `get_model/config/` directory, such as `fetal_region_erythroblast.yaml`. Main configurations are composed of model, finetune, and machine configurations, as well as other task-specific parameters.

### Model Configurations

These are YAML files in the `get_model/config/model/` directory, such as `GETRegionFinetune.yaml` and `GETNucleotideRegionFinetuneExpHiCABC.yaml`. These files specify model-specific parameters and architecture details.

   The common structure of a model configuration typically includes:

   a. **Model Target**: Specifies the Python class to be instantiated for the model.
      ```yaml
      model:
        _target_: get_model.model.model.GETRegionFinetune
      ```

   b. **Model Configuration**: A nested structure under `cfg` that defines various model parameters:
      ```yaml
      cfg:
        num_regions: 900
        num_motif: 283
        embed_dim: 768
        num_layers: 12
        num_heads: 12
        dropout: 0.1
        output_dim: 2
        flash_attn: false
        pool_method: "mean"
      ```

   c. **Component Configurations**: Specific configurations for different parts of the model, such as:
      - `region_embed`: Configuration for region embedding
      - `encoder`: Configuration for the transformer encoder
      - `head_exp`: Configuration for the expression prediction head
      - `motif_scanner`: Configuration for motif scanning (if applicable)
      - `atac_attention`: Configuration for ATAC-seq attention (if applicable)

      Example:
      ```yaml
      region_embed:
        num_regions: ${model.cfg.num_regions}
        num_features: ${model.cfg.num_motif} 
        embed_dim: ${model.cfg.embed_dim}
      
      encoder:
        num_heads: ${model.cfg.num_heads}
        embed_dim: ${model.cfg.embed_dim}
        num_layers: ${model.cfg.num_layers}
        drop_path_rate: ${model.cfg.dropout}
        drop_rate: 0
        attn_drop_rate: 0
        use_mean_pooling: false
        flash_attn: ${model.cfg.flash_attn}
      ```

   d. **Loss Configuration**: Specifies the loss functions and their weights for different components of the model:
      ```yaml
      loss:
        components:
          exp:
            _target_: torch.nn.PoissonNLLLoss
            reduction: "mean"
            log_input: False
        weights:
          exp: 1.0
      ```

   e. **Metrics Configuration**: Defines the evaluation metrics for different model outputs:
      ```yaml
      metrics:
        components:
          exp: ["pearson", "spearman", "r2"]
      ```

   The model configuration files use Hydra's composition and interpolation features. For example, `${model.cfg.num_heads}` refers to the value defined earlier in the same file. This allows for easy parameter sharing and consistency across the model architecture.

   Different model variants (e.g., GETRegionFinetune, GETNucleotideRegionFinetuneExpHiCABC) will have different configurations reflecting their specific architectures and capabilities. For instance, a model that includes Hi-C and ABC predictions will have additional heads and loss components for these tasks.

   When creating or modifying a model, you'll need to ensure that the YAML configuration aligns with the corresponding Python dataclass in `get_model/model/model.py`, such as `GETRegionFinetuneModelConfig` or `GETPretrainModelConfig`.

### Machine Configurations

These are YAML files in the `get_model/config/machine/` directory, such as `pc.yaml`. These configurations specify machine-specific settings, including:

   - `data_path`: The root directory for dataset files
   - `codebase`: The directory containing the project's source code
   - `output_dir`: The directory where output files (e.g., logs, checkpoints) will be saved
   - `num_devices`: The number of GPU devices to use for training
   - `num_workers`: The number of worker processes for data loading
   - `batch_size`: The batch size for training and evaluation
   - `fasta_path`: The path to the FASTA file containing reference genome sequences

   Example `pc.yaml`:
   ```yaml
   data_path: "/home/user/Projects/get_data/"
   codebase: "/home/user/Projects/get_model/"
   output_dir: "/home/user/output"
   num_devices: 1
   num_workers: 4
   batch_size: 32
   fasta_path: "/home/user/data/reference_genome.fa"
   ```

   These machine-specific configurations allow for easy adaptation to different computing environments, such as local workstations, high-performance clusters, or cloud instances. By separating these settings, you can quickly switch between different hardware setups without modifying the core model or dataset configurations.

   You can create multiple machine configuration files (e.g., `pc.yaml`, `cluster.yaml`, `cloud.yaml`) to easily switch between different environments by specifying the appropriate file in your task-specific configuration or via command-line override.

    To switch with command line, use `+machine=cluster` to switch to `cluster.yaml`.
    To switch in task-specific configuration, add `machine/cluster` to the `defaults` list.

### Training Configurations

These are defined in the `TrainingConfig` dataclass in `get_model/config/config.py`. They specify parameters related to the training process:

   - `save_ckpt_freq`: Frequency of saving checkpoints (in epochs)
   - `epochs`: Total number of training epochs
   - `warmup_epochs`: Number of warmup epochs
   - `accumulate_grad_batches`: Number of batches to accumulate gradients over
   - `clip_grad`: Maximum norm of the gradients (if None, no clipping is performed)
   - `use_fp16`: Whether to use 16-bit floating-point precision

   Example usage in a YAML file:
   ```yaml
   training:
     save_ckpt_freq: 1
     epochs: 100
     warmup_epochs: 5
     accumulate_grad_batches: 1
     clip_grad: null
     use_fp16: true
   ```

### Fine-tuning Configurations

These are defined in the `FinetuneConfig` dataclass in `get_model/config/config.py`. They control how the model is fine-tuned:

   - `resume_ckpt`: Path to a checkpoint to resume training from
   - `pretrain_checkpoint`: Whether to use a pretrained checkpoint
   - `checkpoint`: Path to the main model checkpoint
   - `strict`: Whether to strictly enforce that the keys in the checkpoint match the model
   - `model_key`: Key for accessing the model state in the checkpoint dictionary
   - `use_lora`: Whether to use Low-Rank Adaptation (LoRA) for fine-tuning
   - `lora_checkpoint`: Path to a LoRA checkpoint
   - `rename_config`: Configuration for renaming layers when loading a checkpoint
   - `layers_with_lora`: List of layer names to apply LoRA to
   - `patterns_to_freeze`: List of patterns to match layer names that should be frozen during fine-tuning
   - `patterns_to_drop`: List of patterns to match layer names that should be dropped when loading a checkpoint
   - `additional_checkpoints`: List of additional checkpoints to load

   Example usage in a YAML file:
   ```yaml
   finetune:
     pretrain_checkpoint: false
     checkpoint: "/path/to/checkpoint.pth"
     strict: true
     use_lora: false
     layers_with_lora: ["region_embed", "encoder", "head_exp"]
     patterns_to_freeze: ["motif_scanner"]
     patterns_to_drop: []
   ```

These configuration classes provide fine-grained control over the training process and model fine-tuning. They can be customized in the task-specific YAML files or overridden via command-line arguments when running the model.

### Dataset Configurations

Dataset configurations are defined in the `DatasetConfig`, `ReferenceRegionDatasetConfig`, and `RegionDatasetConfig` dataclasses in `get_model/config/config.py`. These configurations control various aspects of data loading and preprocessing.

1. **DatasetConfig**: This is the base configuration for datasets, used primarily for nucleotide-level models.

   Key attributes include:
   - `zarr_dirs`: List of Zarr directory paths containing the dataset
   - `n_peaks_lower_bound` and `n_peaks_upper_bound`: Range for the number of peaks to consider
   - `max_peak_length`: Maximum length of a peak
   - `leave_out_celltypes` and `leave_out_chromosomes`: Specify data to exclude from training
   - `mask_ratio`: Ratio for masking input data (used in pretraining)
   - `dataset_size` and `eval_dataset_size`: Size of the training and evaluation datasets

2. **ReferenceRegionDatasetConfig**: Extends `DatasetConfig` with additional parameters specific to reference region models.

   Additional attributes include:
   - `reference_region_motif`: Configuration for reference region motif analysis
   - `quantitative_atac`: Whether to use quantitative ATAC-seq data
   - `sampling_step`: Step size for sampling reference regions
   - `leave_out_motifs`: Motifs to exclude from the dataset

3. **RegionDatasetConfig**: Used for region-specific models, with attributes tailored for regional analysis.

   Key attributes include:
   - `root`: Root directory for the dataset
   - `metadata_path`: Path to the metadata file
   - `num_region_per_sample`: Number of regions to consider per sample
   - `data_type`: Type of data (e.g., "fetal")

Example usage in a YAML file:
```yaml
dataset:
    zarr_dirs:
        - "/path/to/zarr/dir1"
        - "/path/to/zarr/dir2"
    n_peaks_lower_bound: 5
    n_peaks_upper_bound: 10
    leave_out_celltypes: "Astrocyte"
    leave_out_chromosomes: "chr4,chr14"
    mask_ratio: 0.15
    dataset_size: 40960
    eval_dataset_size: 4096
```

### Run Configurations

Run configurations are defined in the `RunConfig` dataclass in `get_model/config/config.py`. These configurations specify high-level parameters for the experimental run.

Key attributes of `RunConfig` include:
- `project_name`: Name of the project (used for logging and organization)
- `run_name`: Specific name for this run or experiment

Example usage in a YAML file:
```yaml
run:
    project_name: "GETRegionFinetuneV1_Erythroblast"
    run_name: "debug"
```

The run configuration is typically used in conjunction with logging and experiment tracking tools like Weights & Biases (wandb). In the `setup_wandb` function in `get_model/utils.py`, these values are used to initialize the wandb logger:

```python
def setup_wandb(cfg):
    wandb_logger = WandbLogger(
    name=cfg.run.run_name,
    project=cfg.run.project_name,
    entity="get-v3",
    )
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    return wandb_logger
```

This setup allows for easy tracking and comparison of different experimental runs, with each run having a unique identifier within the project.



## Conclusion

The Hydra-based configuration system provides a flexible and powerful way to manage different experimental setups. By understanding the structure of the configuration files and how to override values, you can easily adapt the system to your specific needs without modifying the core code.