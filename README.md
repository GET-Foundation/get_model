# GET: General Expression Transformer

A foundation model of transcription across human cell types.

This repository contains the official implementation of the model described in our paper: https://www.nature.com/articles/s41586-024-08391-z.


## Table of Contents

- [GET: General Expression Transformer](#get-general-expression-transformer)
  - [Table of Contents](#table-of-contents)
  - [Tutorials](#tutorials)
  - [Installation-Pip](#installation-pip)
  - [Installation-Conda](#installation-conda)
  - [Installation-Docker/Singularity](#installation-dockersingularity)
  - [Model Architecture](#model-architecture)
  - [Command line interface](#command-line-interface)
  - [Configuration](#configuration)
  - [Contributing](#contributing)
  - [License](#license)
  - [Citation](#citation)
  - [Contact](#contact)

## Tutorials
- [Data processing](tutorials/prepare_pbmc.ipynb)
- [Finetune & Interpretation](tutorials/finetune_pbmc.ipynb) 
- [Moitf -> ATAC prediction](tutorials/predict_atac.ipynb) (just for demo, optional)
- [Continue pretrain](tutorials/pretrain_pbmc.ipynb) (just for demo, optional)

Note that `Motif -> ATAC prediction` tutorial has been tested on a Macbook Pro M4 Pro with MPS accelaration. It seems that the speed for training and validation iteration is close to a RTX3090; 
However, some ops used in the metric calculation (Pearson/Spearman/R^2) was not accelarated, making the speed a bit inferior. 

## Installation-Pip
If you just need the model and analysis package. You can install with pip. However, note that the R package `pcalg` is required for the causal analysis and will not be available if you don't install it manually.
```bash
pip install git+https://github.com/GET-Foundation/get_model.git@master
```

## Installation-Conda

You can use conda/mamba for environment setup. The `env.yml` will install the following packages:
```
- get_model: main model package
  - [gcell](https://github.com/GET-Foundation/gcell): the analysis interface and demo backend
  - [genomespy](https://github.com/fuxialexander/genomespy): an interactive genome browser within jupyter notebook
- wget: in case you don't have it
- gawk: GNU awk, in case you don't have it
- bedtools
- htslib
- r-pcalg: for causal discovery of motif-motif interaction
- scanpy: for single cell analysis (optional, required just for tutorial).
- snapatac2: for scATAC-seq analysis (optional, required just for tutorial).  
```

If you don't want all of them, you can install just the get_model package with pip.
Note that if you have problem installing the conda/mamba environment, edit (temporarily) your CONDARC to remove `channel_priority: strict` 
```bash
mamba env create -f env.yml
```
If you are on Mac OS and Apple Silicon, you can try to run the following:
```bash
mamba env create -f env_osx.yml
# install brew if you haven't
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# install R with brew
brew install r
# install pcalg with bioconductor
R -e 'if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager", repos="https://cloud.r-project.org"); BiocManager::install("pcalg")'
# test pcalg loads within R
R -e 'library(pcalg); cat("pcalg loaded successfully\n")'
```

## Installation-Docker/Singularity

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
import gcell
```

If you are using vscode or cursor as code editor, you can open a tunnel from inside the singularity / docker
```bash
singularity exec --nv get.sif /bin/bash
# then 
code tunnel
# or 
cursor tunnel
```
This enable you to use your local Cursor.app or VSCode.app and all the Copilot/Jupyter/Debugger stuff to access the environment inside the container. You can even access it from your browser.


## Model Architecture

GET uses a transformer-based architecture with several key components:
- Region Embedding
- Transformer Encoder
- Task-specific heads (Expression, ATAC, etc.)

In the future, nucleotide modeling and more modality (e.g. Hi-C, ChIP-seq) will be incorporated. All variation of model will be constructed in a modular and composable way.
For more details, check out this [Schematic](https://fuxialexander.github.io/get_model/model.html) or [Model Architecture](tutorials/Model%20Customization.md).

## Command line interface

We use [Hydra](https://hydra.cc) for configuration management and command line interface. Hydra provides a flexible way to configure and run experiments by:

- Managing hierarchical configurations through YAML files
- Enabling command line overrides of config values
- Supporting multiple configuration groups
- Allowing dynamic composition of configurations

See the example debug scripts in `get_model/debug/` for how to write a command line training script.

To run a basic training job in command line:
```bash
python get_model/debug/debug_run_region.py --config-name finetune_tutorial stage=fit
```

## Configuration

GET uses Hydra for configuration management. Key configuration files:

- Base config: `get_model/config/config.py`
- Model configs: `get_model/config/model/*.yaml`
- Dataset configs: `get_model/config/dataset/*.yaml`

See [Configuration Guide](tutorials/Configuration.md) for more details.

## Contributing

We use `hatch` to manage the development environment.

```bash
hatch env create
```


## License

This project is licensed under the CC BY-NC 4.0 License. For commercial use, please contact us.

## Citation

If you use GET in your research, please cite our paper:

A foundation model of transcription across human cell types. Nature (2024). https://doi.org/10.1038/s41586-024-08391-z


## Contact

For questions or support, please open an issue or contact [fuxialexander@gmail.com](mailto:fuxialexander@gmail.com).
