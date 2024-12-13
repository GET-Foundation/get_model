# GET: Gene Expression Transformer

GET: a foundation model of transcription across human cell types

## Table of Contents

- [GET: Gene Expression Transformer](#get-gene-expression-transformer)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Model Architecture](#model-architecture)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Configuration](#configuration)
  - [Contributing](#contributing)
  - [License](#license)
  - [Citation](#citation)
  - [Contact](#contact)

## Installation

For conda/mamba installation, checkout scripts/setup_env.sh to setup the environment. Note that if you have problem installing the conda/mamba environment, edit (temporarily) your CONDARC to remove `channel_priority: strict` 
```bash
bash scripts/setup_env.sh /path/to/project/root
```

Alternatively, a docker image is provided for running the code. 

```bash
docker pull fuxialexander/get_model:latest
```
This start a bash shell in the container by default
```bash
docker run -it -v /home/xf2217:/home/xf2217 fuxialexander/get_model 
```

You can also start a jupyter notebook server in the container and access it from your host machine on port 8888

```bash
docker run --entrypoint /opt/conda/bin/jupyter -it -p 8888:8888 -v /home/xf2217:/home/xf2217 fuxialexander/get_model notebook --allow-root --ip 0.0.0.0 --no-browser --NotebookApp.token='' --NotebookApp.password='' # add password if you want
```
then you can access the jupyter notebook server at `http://localhost:8888`, in VSCode, you can open a jupyter notebook and select kernel to use existing jupyter server, and put in `http://localhost:8888` as the server URL.

You can also directly acess the python with the following command
```bash
docker run -it -v /home/xf2217:/home/xf2217 fuxialexander/get_model /opt/conda/bin/python /some/script/to/run.py
```

For singularity installation, you can pull the docker image and convert it to singularity image. 
```bash
# module load singularity if needed 
singularity pull get.sif docker://fuxialexander/get_model:latest
# start a jupyter notebook server
singularity exec --nv get.sif env JUPYTER_CONFIG_DIR=/tmp/.jupyter /opt/conda/bin/jupyter notebook --allow-root --ip 0.0.0.0 --no-browser --NotebookApp.token='' --NotebookApp.password=''
# or directly access python
singularity exec --nv get.sif /opt/conda/bin/python
```
then test if cuda is avaliable and whether package is installed correctly:
```python
import torch
torch.cuda.is_available()
import get_model
import atac_rna_data_processing
```

## Quick Start

We provide a tutorial on how to prepare the data, finetune the model, and do interpretation analysis [here](tutorials/full_v1_pipeline.py).

To run a basic training job in command line:
```bash
python get_model/debug/debug_run_region.py --config-name finetune_tutorial stage=fit
```

## Model Architecture

GET uses a transformer-based architecture with several key components:
- Motif Scanner
- ATAC Attention
- Region Embedding
- Transformer Encoder
- Task-specific heads (Expression, Hi-C, etc.)

For more details, check out this [Schematic](https://fuxialexander.github.io/get_model/model.html) or [Model Architecture](tutorials/Model%20Customization.md).

## Training

To fine-tune a pre-trained model:

See [Fine-tuning Tutorial](tutorials/Finetune.md) for more information.

## Evaluation

To evaluate a trained model:
```bash
python get_model/debug/debug_run_region.py --config-name finetune_tutorial stage=validate
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
