FROM mambaorg/micromamba:2-cuda12.5.1-ubuntu24.04 

COPY --chown=$MAMBA_USER:$MAMBA_USER env.yml /tmp/env.yaml

RUN micromamba install -y git openssh gcc gxx libgcc-ng -n base -c conda-forge

RUN mkdir -p /home/$MAMBA_USER/.ssh

ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN ssh-keyscan www.github.com >> /home/$MAMBA_USER/.ssh/known_hosts

RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes


# install git with mamba
RUN micromamba clean --all --yes
# change back to mamba user
USER $MAMBA_USER

# Remove cached packages
# Pip cache
RUN rm -rf /home/$MAMBA_USER/.cache/pip


# set user to root and activate mamba
USER root

# install vscode
RUN wget "https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-x64" -O "/tmp/vscode.deb" && \
    apt-get update && apt-get install -y /tmp/vscode.deb && \
    apt-get install -y wget unzip tar && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/vscode.deb

# install aws cli
RUN wget "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -O "/tmp/awscliv2.zip" && \
    cd /tmp/ && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf /tmp/awscliv2.zip && \
    rm -rf /tmp/aws


# install cursor
RUN wget "https://api2.cursor.sh/updates/download-latest?os=cli-alpine-x64" -O "/tmp/cursor.tar.gz" && \
    tar -xvf /tmp/cursor.tar.gz && \
    mv cursor /usr/bin/cursor && \
    rm -rf /tmp/cursor.tar.gz

ENV PATH="/opt/conda/bin/:$PATH:/usr/bin:/usr/local/bin"

