# Set the root directory where the project will be set up
ROOT_DIR=$1

# Check if the specified directory does not exist, and if so, create the necessary directories
if [ ! -d $ROOT_DIR ]; then
    mkdir -p $ROOT_DIR  # Create the root directory if it does not exist
fi
# if conda env exist, remove and recreate
if [ -d $ROOT_DIR/mambaforge ]; then
    rm -rf $ROOT_DIR/mambaforge
fi
# Navigate into the project root directory
cd $ROOT_DIR

# Define the directory where the codebase will be located
CODEBASE_DIR=$ROOT_DIR/get_model

# Clone the get_model repository into the codebase directory, if exist, pull the latest code
if [ -d $CODEBASE_DIR ]; then
    cd $CODEBASE_DIR
    git pull
else
    git clone --recurse-submodules --shallow-submodules -b refactor_with_hydra git@github.com:fuxialexander/get_model.git
    cd $CODEBASE_DIR
fi

# Switch to a specific branch of the cloned repository
git checkout refactor_with_hydra

# Create a new mamba environment based on the provided environment.yml file
mamba env create -f ${CODEBASE_DIR}/environment.yml -p ${ROOT_DIR}/mambaforge/get_started

# Activate the newly created mamba environment
source activate ${ROOT_DIR}/mambaforge/get_started

# Return to the codebase directory and install the package in editable mode
cd $CODEBASE_DIR
pip install -e .

# Clone the caesar repository into the project directory and install it, if exist, pull the latest code
cd $ROOT_DIR
if [ -d $ROOT_DIR/caesar ]; then
    cd $ROOT_DIR/caesar
    git pull
else
    git clone git@github.com:fuxialexander/caesar.git
fi

cd $ROOT_DIR/caesar
pip install -e .

# Clone the atac_rna_data_processing repository into the project directory and install it, if exist, pull the latest code
cd $ROOT_DIR
if [ -d $ROOT_DIR/atac_rna_data_processing ]; then
    cd $ROOT_DIR/atac_rna_data_processing
    git pull
else
    git clone git@github.com:fuxialexander/atac_rna_data_processing.git
fi 

cd $ROOT_DIR/atac_rna_data_processing
pip install -e .

pip install git+https://github.com/pyranges/pyranges@master
pip install git+https://github.com/cccntu/minLoRA.git@main
# Return to the codebase directory
cd $CODEBASE_DIR
