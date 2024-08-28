# GET: Gene Expression Transformer

GET: a foundation model of transcription across human cell types

## Table of Contents

- [GET: Gene Expression Transformer](#get-gene-expression-transformer)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Data Preparation](#data-preparation)
  - [Model Architecture](#model-architecture)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Configuration](#configuration)
  - [Contributing](#contributing)
  - [License](#license)
  - [Citation](#citation)
  - [Contact](#contact)

## Installation
```bash
pip install -e .
```

## Quick Start

To run a basic training job:
```bash
python get_model/debug/debug_run_ref_region_hic_oe.py --config-name eval_k562_fetal_ref_region_k562_hic_oe stage=fit
```
## Data Preparation

GET requires preprocessed ATAC-seq, RNA-seq, and optionally Hi-C data. See the [data preprocessing tutorial](tutorials/Dataloader.md) for detailed instructions.

## Model Architecture

GET uses a transformer-based architecture with several key components:
- Motif Scanner
- ATAC Attention
- Region Embedding
- Transformer Encoder
- Task-specific heads (Expression, Hi-C, etc.)

For more details, see [Model Architecture](tutorials/Model%20Customization.md).

## Training

To fine-tune a pre-trained model:

See [Fine-tuning Tutorial](tutorials/Finetune.md) for more information.

## Evaluation

To evaluate a trained model:
```bash
python get_model/debug/debug_run_ref_region_hic_oe.py --config-name eval_k562_fetal_ref_region_k562_hic_oe stage=validate
```

## Configuration

GET uses Hydra for configuration management. Key configuration files:

- Base config: `get_model/config/config.py`
- Model configs: `get_model/config/model/*.yaml`
- Dataset configs: `get_model/config/dataset/*.yaml`

See [Configuration Guide](tutorials/Configuration.md) for more details.

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under the CC BY-NC 4.0 License.

## Citation

If you use GET in your research, please cite our paper:

## Contact

For questions or support, please open an issue or contact [fuxialexander@gmail.com](mailto:fuxialexander@gmail.com).
