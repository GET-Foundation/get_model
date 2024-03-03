# Use a base image with mamba installed, assuming a Linux distribution compatible with your setup
FROM mambaorg/micromamba:latest

# Set the working directory inside the container to a more conventional path
WORKDIR /GET_STARTED

# Clone the necessary repositories and set up the environments
RUN git clone git@github.com:fuxialexander/get_model.git && \
    cd get_model && \
    git checkout finetune-with-atac && \
    mamba env create -f environment.yml -p /opt/conda/envs/get_started && \
    echo "source activate /opt/conda/envs/get_started" > ~/.bashrc && \
    pip install -e . && \
    cd /GET_STARTED && \
    git clone git@github.com:fuxialexander/caesar.git && \
    cd caesar && \
    pip install -e . && \
    cd /GET_STARTED && \
    git clone git@github.com:fuxialexander/atac_rna_data_processing.git && \
    cd atac_rna_data_processing && \
    pip install -e .

# Set the final working directory
WORKDIR /GET_STARTED
