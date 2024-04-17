# Use a base image with mamba installed, assuming a Linux distribution compatible with your setup
FROM mambaorg/micromamba:latest

# Set the working directory inside the container to a more conventional path
WORKDIR /GET_STARTED
# root user is required to install packages
USER root
RUN apt-get update && apt-get install -y git
USER $MAMBA_USER

# map the repository to the container
COPY . /GET_STARTED/get_model

WORKDIR /GET_STARTED/get_model
# Clone the necessary repositories and set up the environments
RUN mamba env create -f environment.yml -p /opt/conda/envs/get_started && \
    echo "source activate /opt/conda/envs/get_started" > ~/.bashrc && \
    pip install . && \
    cd /GET_STARTED/get_model/modules/caesar && \
    pip install . && \
    cd /GET_STARTED/get_model/modules/atac_rna_data_processing && \
    pip install .

# Set the final working directory
WORKDIR /GET_STARTED
