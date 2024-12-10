# Fine-Tuning a Model for ATAC-Seq and RNA-Seq Data

This tutorial walks you through the process of fine-tuning a model using single-cell ATAC-seq data or bulk-ATAC-seq data and RNA-seq gene expression data. We will cover data preprocessing, setting up configurations, running the fine-tuning, and performing downstream analysis.

## Step 1: Pre-process the Data

You need ATAC-seq data in the form of a fragment file (for single-cell data) or a `BAM` file (for bulk data). For Region-based model in the original GET paper, you can run MACS peak calling on the fragment file to acquire a peak file. For RNA-seq, you will need a list of `CSV` files denoting gene expression or a single `CSV` file. Checkout the `astrocyte.atac.bed` and `astrocyte.rna.csv` for the file required in `full_v1_pipeline.py`.

The output will be a `zarr` file containing both ATAC-seq pseudoblock data and RNA-seq expression values.

Refer to the specific data preprocessing tutorial for detailed steps. 

## Step 2: Configuration Settings

Depending on your data input setting, the configuration might vary. Below is a high-level overview of the required configurations.

### Example Configuration Structure

1. **Base Configuration**: General settings and paths.
2. **Model Configuration**: Specific settings for the model architecture.
3. **Finetune Configuration**: Settings related to the fine-tuning process.
4. **Dataset Configuration**: Details about the dataset being used.

Refer to `get_model/config` for specific config examples.

### Sample YAML Configuration Overview

A sample configuration file can be found in get_model/config/eval_k562_fetal_ref_region_k562_hic_oe.yaml (and many others).
```yaml
defaults:
  - base_ref_region_config # this is the base configuration for reference region settings, defined in get_model/config/config.py
  - model/GETRegionFinetuneHiCOE@_here_ # this is the model configuration which means we use all the config in get_model/config/model/GETRegionFinetuneHiCOE.yaml and put them directly here
  - machine/pc # this is the machine config, which is suppose to specify machine specific settings like data/codebase paths, number of workers and GPUs, etc
  - finetune/v1_finetune@finetune # this is the finetune config, which is suppose to specify the checkpoint to use for finetuning and more
  - dataset/k562_hic_oe@dataset # this is the dataset config, which is suppose to specify the paths to the data, settings for loading and preprocessing data, and other dataset specific parameters
  - _self_ # this is a required component

# other configs
training:
  save_ckpt_freq: 1 # save checkpoint every epoch
  epochs: 100 # number of epochs to train
  warmup_epochs: 1 # number of warmup epochs
  accumulate_grad_batches: 1 # number of batches to accumulate gradients
  clip_grad: null # gradient clipping value
  use_fp16: false # use half precision training

dataset:
  quantitative_atac: true # use quantitative ATAC-seq data
  sampling_step: 100 # sampling step for the data
  mask_ratio: 0 # mask ratio for the data

optimizer:
  lr: 0.0001 # learning rate
  min_lr: 0.000001 # minimum learning rate
  weight_decay: 0.05 # weight decay
  opt: "adamw" # optimizer
  opt_eps: 1e-8 # optimizer epsilon
  opt_betas: [0.9, 0.999] # optimizer betas
 
run:
  project_name: "GETRegionFinetune_k562_cage" # wandb project name
  run_name: "debug" # wandb run name
```

**Model Configuration**

This config defines the model architecture, layers, and other hyperparameters. Use the appropriate model config based on your requirements.

**Finetune Configuration**

Defines the checkpoint to use, model key, and layer-specific settings for fine-tuning.

**Dataset Configuration**

Specifies paths to the data, settings for loading and preprocessing data, and other dataset-specific parameters.

## Step 3: Run the Fine-Tuning

To fine-tune the model, use the following command:

```bash
python get_model/debug/debug_run_ref_region_hic_oe.py --config-name eval_k562_fetal_ref_region_k562_hic_oe stage=fit
```

You can override specific configurations via command line:

```bash
python get_model/debug/debug_run_ref_region_hic_oe.py --config-name eval_k562_fetal_ref_region_k562_hic_oe stage=fit dataset.peak_count_filter=10 dataset.reference_region_motif.motif_scaler=1.3 machine.num_workers=4 machine.batch_size=8
```

## Step 4: Downstream Analysis

Once fine-tuning is complete, gather the checkpoints and prepare scripts for further analysis. 

