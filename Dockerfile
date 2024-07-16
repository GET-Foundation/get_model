FROM mambaorg/micromamba:latest


COPY --chown=$MAMBA_USER:$MAMBA_USER environment_for_docker.yml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1

# install git with mamba
RUN micromamba install -y git openssh -c conda-forge

# change back to mamba user
USER $MAMBA_USER

RUN mkdir -p /home/$MAMBA_USER/.ssh
RUN ssh-keyscan www.github.com >> /home/$MAMBA_USER/.ssh/known_hosts
# copy the codebase to the container
COPY --chown=$MAMBA_USER:$MAMBA_USER . /home/$MAMBA_USER/get_model

# Install the package in editable mode
RUN cd /home/$MAMBA_USER/get_model && \
    pip install -e .

# Clone the caesar repository into the project directory and install it
RUN cd /home/$MAMBA_USER/get_model/modules/ && \
    cd caesar && \
    pip install -e .

# Clone the atac_rna_data_processing repository into the project directory and install it
RUN cd /home/$MAMBA_USER/get_model/modules/ && \
    cd atac_rna_data_processing && \
    pip install -e .

# Install additional dependencies
RUN pip install cython==3.0.8 einops==0.7.0 hic-straw==1.3.1 scanpy==1.9.8 MOODS-python hydra-core lightning ghostscript pytest && \
    pip install git+https://github.com/pyranges/pyranges@master && \
    pip install git+https://github.com/cccntu/minLoRA.git@master

# Set the working directory to the codebase directory
WORKDIR /home/$MAMBA_USER/get_model/get_model