# Set the root directory where the project will be set up
USER=$1
ROOT_DIR=/pmglocal/${USER}/GET_STARTED
DATA_DIR=/burg/pmg/users/xf2217/get_data

# Check if the specified directory does not exist, and if so, create the necessary directories
if [ ! -d $ROOT_DIR ]; then
    mkdir -p $ROOT_DIR  # Create the root directory if it does not exist
fi

# Navigate into the project root directory
cd $ROOT_DIR

# Define the directory where the codebase will be located
CODEBASE_DIR=$ROOT_DIR/get_model

# Clone the get_model repository into the codebase directory
git clone git@github.com:fuxialexander/get_model.git
cd $CODEBASE_DIR

# Switch to a specific branch of the cloned repository
git checkout refactor_with_hydra

# Create a new mamba environment based on the provided environment.yml file
mamba env create -f ${CODEBASE_DIR}/environment.yml -p ${ROOT_DIR}/mambaforge/get_started

# Activate the newly created mamba environment
source activate ${ROOT_DIR}/mambaforge/get_started

# Copy data from a specified location to the project directory
#cp -r $DATA_DIR ${ROOT_DIR}/get_data

# Navigate to the data directory and decompress all .tar files
cd ${ROOT_DIR}/get_data
#for f in *.tar; do tar -xvf $f; done

# Return to the codebase directory and install the package in editable mode
cd $CODEBASE_DIR
pip install -e .

# Clone the caesar repository into the project directory and install it
cd $ROOT_DIR
git clone git@github.com:fuxialexander/caesar.git
cd $ROOT_DIR/caesar
pip install -e .

# Clone the atac_rna_data_processing repository into the project directory and install it
cd $ROOT_DIR
git clone git@github.com:fuxialexander/atac_rna_data_processing.git
cd $ROOT_DIR/atac_rna_data_processing
pip install -e .

pip install git+https://github.com/pyranges/pyranges@master
pip install git+https://github.com/cccntu/minLoRA.git@master
# Return to the codebase directory
cd $CODEBASE_DIR
