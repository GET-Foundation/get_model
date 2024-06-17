# README: Customizing and Constructing Models Using the GET Framework

This document provides technical specifications and guidelines for customizing and constructing models using the GET framework. The framework is designed to be modular, flexible, and easily extensible, allowing you to create a wide variety of models tailored to your specific needs.

## Overview

The GET framework includes several base classes and configurations that serve as building blocks for creating complex models. The primary components include:

- **BaseGETModel**: A base class for all models, providing common functionalities such as loss calculation, metric evaluation, weight initialization, and data handling.
- **BaseGETModelConfig**: A configuration class for specifying model parameters.
- **Module Components**: Predefined modules such as `MotifScanner`, `SplitPool`, `GETTransformer`, and others, which can be combined to create new models.
- **Loss and Metrics**: Custom loss functions and metrics tailored to specific tasks.

## Creating a New Model

To create a new model, you typically need to follow these steps:

1. **Define a Configuration Class**: Specify the parameters for your model components by creating a dataclass that inherits from `BaseGETModelConfig`.

2. **Implement the Model Class**: Inherit from `BaseGETModel` and define the model architecture in the `__init__` method, the forward pass logic in `forward`, input processing in `get_input`, loss preparation in `before_loss`, and generate dummy data in `generate_dummy_data`.

3. **Configure Model Parameters**: Create a YAML file in the `get_model/config/model/` directory to specify the model parameters. The config should also contain a `_target_` key pointing to the corresponding model class for Hydra to load the model.

4. **Configure Dataset**: Define a dataset config in the `get_model/config/dataset/` directory, specifying the dataset to use for training. 

5. **Configure Run Settings**: Check the overall run config in the `get_model/config/` directory, specifying settings like the model, dataset, machine, finetune parameters etc.

6. **Prepare Run Script**: Make sure the script to run the model (e.g. `get_model/debug*`) is configured correctly to use the corresponding configs. If implementing a new model class that involves new training procedures, also look into the run scripts defined in `get_model.run_ref_region` and make necessary modifications.

Here are more details on each step:

### Step 1: Define a Configuration Class

Create a configuration dataclass inheriting from `BaseGETModelConfig`. This class will hold the configurations for the various components of your model. For example:

```python
from dataclasses import dataclass, field

@dataclass 
class MyModelConfig(BaseGETModelConfig):
    motif_scanner: MotifScannerConfig = field(default_factory=MotifScannerConfig)  
    atac_attention: SplitPoolConfig = field(default_factory=SplitPoolConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    head_exp: ExpressionHeadConfig = field(default_factory=ExpressionHeadConfig)
    head_hic: ContactMapHeadConfig = field(default_factory=ContactMapHeadConfig)
```

### Step 2: Implement the Model Class

Implement the model class inheriting from `BaseGETModel`. Define the following key methods:

- `__init__`: Initialize the model components based on the configuration 
- `get_input`: Process the input batch and return a dictionary of inputs to the model
- `forward`: Define the forward pass logic
- `before_loss`: Prepare the predictions and observations for loss calculation
- `generate_dummy_data`: Generate dummy data for sanity checks

Here's an example:

```python
class MyModel(BaseGETModel):
    def __init__(self, cfg: MyModelConfig):
        super().__init__(cfg)
        self.motif_scanner = MotifScanner(cfg.motif_scanner)
        self.atac_attention = SplitPool(cfg.atac_attention) 
        self.encoder = GETTransformer(**cfg.encoder)
        self.head_exp = ExpressionHead(cfg.head_exp)
        self.head_hic = ContactMapHead(cfg.head_hic)
        self.proj = nn.Linear(cfg.motif_scanner.num_motif, cfg.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.apply(self._init_weights)

    def get_input(self, batch, perturb=False):
        sample_peak_sequence = batch['sample_peak_sequence'] 
        sample_track = batch['sample_track']
        chunk_size = batch['chunk_size']
        n_peaks = batch['n_peaks'] 
        max_n_peaks = batch['max_n_peaks']
        return {
            'sample_peak_sequence': sample_peak_sequence,
            'sample_track': sample_track,
            'chunk_size': chunk_size,
            'n_peaks': n_peaks,
            'max_n_peaks': max_n_peaks,
        }

    def forward(self, sample_peak_sequence, sample_track, chunk_size, n_peaks, max_n_peaks):
        x = self.motif_scanner(sample_peak_sequence)
        x = x.permute(0, 2, 1)
        x = self.atac_attention(x, chunk_size, n_peaks, max_n_peaks) 
        x = self.proj(x)

        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x, _ = self.encoder(x)
        x = x[:, 1:]
        exp = nn.Softplus()(self.head_exp(x))
        hic = self.head_hic(x) 
        return exp, hic

    def before_loss(self, output, batch):  
        exp, hic = output
        pred = {'exp': exp, 'hic': hic}
        obs = {'exp': batch['exp_label'], 'hic': batch['hic_matrix']}
        return pred, obs

    def generate_dummy_data(self):
        B, R, L = 2, 1, 2000  
        return {
            'sample_peak_sequence': torch.randint(0, 4, (B, R * L, 4)).float(),
            'sample_track': torch.randn(B, R, 283).float(),
            'chunk_size': torch.tensor([R]),
            'n_peaks': torch.tensor([R]),  
            'max_n_peaks': torch.tensor([R]),
        }
```

### Step 3: Configure Model Parameters 

Create a YAML file in the `get_model/config/model/` directory to specify the parameters for your model. The config should also contain a `_target_` key pointing to the corresponding model class for Hydra to load the model. For example:

```yaml
model:
  _target_: get_model.model.MyModel
  cfg:
    num_regions: 200
    num_motif: 283
    embed_dim: 768
    num_layers: 12
    num_heads: 12 
    dropout: 0.1
    motif_scanner:
      num_motif: 128
      include_reverse_complement: false
    atac_attention:  
      motif_dim: 128
      pool_method: "sum"
    encoder:
      num_heads: ${model.cfg.num_heads}
      embed_dim: ${model.cfg.embed_dim}
    head_exp:
      embed_dim: ${model.cfg.embed_dim}
      output_dim: 1
    head_hic:  
      embed_dim: ${model.cfg.embed_dim}
      output_dim: 1
```

### Step 4: Configure Dataset

Define a dataset config in the `get_model/config/dataset/` directory, specifying the dataset to use for training. For example: 

```yaml
zarr_dirs: 
  - "my_dataset.zarr"
keep_celltypes: "CellTypeA"  
leave_out_celltypes: null
leave_out_chromosomes: "chr14"
peak_name: "peaks"
dataset_size: 40960  
eval_dataset_size: 512
```

### Step 5: Configure Run Settings

Check the overall run config in `get_model/config/` directory, specifying settings like the model, dataset, machine, finetune parameters etc. For example:

```yaml
defaults:
  - model/MyModel@_here_  
  - machine/server
  - finetune/finetune@finetune
  - dataset/my_dataset@dataset
  - _self_

assembly: "hg38"  

task:
  layer_names: []
  test_mode: "predict" 

training:
  epochs: 100
  warmup_epochs: 1 

dataset:
  quantitative_atac: true
  sampling_step: 10

optimizer:  
  lr: 0.001
  weight_decay: 0.05

finetune:
  use_lora: false
  checkpoint: "/path/to/checkpoint.pth"
```

### Step 6: Prepare Run Script

Make sure the script to run the model (e.g. `get_model/debug*`) is configured correctly to use the corresponding configs:

```python
@hydra.main(config_path="../config", config_name="my_run_config")  
def main(cfg: Config):
    run(cfg)
```

If implementing a new model class that involves new training procedures, also look into the run scripts defined in `get_model.run_ref_region` (among others like `run`, `run_region`) and make necessary modifications.

### Tips and Best Practices

- Leverage existing model components whenever possible to promote code reuse and maintainability.
- Be consistent with naming conventions for configurations, models, and modules.
- Use informative docstrings to document your code.
- Thoroughly test your models with different inputs and configurations.
- Consider using dataclasses for cleaner and more readable configuration management.
- Experiment with different model architectures, loss functions, and metrics to find the best combination for your task.
- Monitor training progress and perform early stopping if necessary to prevent overfitting.
- Use version control (e.g., Git) to track changes and collaborate with others.

## Model Loading Process

The GET framework provides a flexible and modular approach to loading model weights and checkpoints. The `get_model` method in the `RegionLitModel` class is responsible for instantiating the model based on the provided configuration and loading the checkpoints. Here's how the model loading process works:

1. **Instantiating the Model**: The model is instantiated using the `instantiate` function from the Hydra library, which creates an instance of the model class specified in the configuration (`self.cfg.model`).

2. **Loading the Main Checkpoint**: If a main checkpoint is specified in the configuration (`self.cfg.finetune.checkpoint`), it is loaded using the `load_checkpoint` function. The loaded checkpoint is then processed using the `extract_state_dict` function to extract the actual model state dictionary, and the `rename_state_dict` function to rename the keys based on the provided `rename_config`. If the checkpoint contains LoRA (Low-Rank Adaptation) parameters and `self.cfg.finetune.use_lora` is set to `True`, the LoRA layers are added to the model using the `add_lora_by_name` function. Finally, the loaded state dictionary is used to initialize the model's weights using the `load_state_dict` function.

3. **Loading Additional Checkpoints**: If additional checkpoints are specified in the configuration (`self.cfg.finetune.additional_checkpoints`), each checkpoint is loaded, processed, and used to update the model's weights in a similar manner to the main checkpoint.

4. **Loading LoRA Parameters**: If `self.cfg.finetune.use_lora` is set to `True`, the LoRA parameters are loaded based on the current stage (`self.cfg.stage`). For the 'fit' stage (training), the LoRA checkpoint is loaded, extracted, renamed, and used to update the model's weights. For the 'validate' and 'predict' stages, a similar process is followed.

5. **Freezing Layers**: The `freeze_layers` function is called to freeze specific layers of the model based on the provided `patterns_to_freeze` and `invert_match` parameters. This is useful for fine-tuning specific parts of the model while keeping others frozen.

The model loading process is supported by several utility functions:

- `extract_state_dict`: Extracts the actual model state dictionary from the loaded checkpoint, handling cases where the state dictionary is nested within the checkpoint dictionary.
- `rename_state_dict`: Renames the keys of the state dictionary based on the provided `rename_config`, allowing for flexibility in matching and replacing patterns in the keys.
- `load_checkpoint`: Loads the checkpoint from the specified path, supporting both URLs and local files. It also allows for selecting a specific key within the checkpoint dictionary using the `model_key` parameter.
- `load_state_dict`: Loads the state dictionary into the model, with support for dropping specific keys based on the provided `patterns_to_drop` parameter.

By leveraging this model loading process, the GET framework enables users to easily load and initialize models from checkpoints, handle LoRA parameters, rename state dictionary keys, and freeze specific layers for fine-tuning.