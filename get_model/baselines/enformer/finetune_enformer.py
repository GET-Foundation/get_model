import tensorflow as tf
# Make sure the GPU is enabled 
assert tf.config.list_physical_devices('GPU')
import glob
import json
import functools
import pandas as pd
import os 
from tqdm import tqdm
import numpy as np
import argparse
import pathlib

import enformer
from utils import *


def main(args):
    np.random.seed(42)
    EXTENDED_SEQ_LENGTH = 393_216
    SEQ_LENGTH = 196_608
    inputs = np.array(np.random.random((1, EXTENDED_SEQ_LENGTH, 4)), dtype=np.float32)
    inputs_cropped = enformer.TargetLengthCrop1D(SEQ_LENGTH)(inputs)
    


    # enformer_model = enformer.Enformer()
    # checkpoint = tf.train.Checkpoint(module=enformer_model)
    # latest = tf.train.latest_checkpoint(checkpoint_path)


def copy_checkpoint_to_local():
    checkpoint_gs_path = 'gs://dm-enformer/models/enformer/sonnet_weights/*'
    checkpoint_path = '/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/ckpts'
    pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    # Copy checkpoints from GCS to temporary directory.
    # This will take a while as the checkpoint is ~ 1GB.
    for file_path in tf.io.gfile.glob(checkpoint_gs_path):
        file_name = os.path.basename(file_path)
        tf.io.gfile.copy(file_path, f'{checkpoint_path}/{file_name}', overwrite=True)
    

if __name__ == '__main__':  
    # argparse = argparse.ArgumentParser()
    # # argparse.add_argument('--organism', type=str, required=True)
    # # argparse.add_argument('--subset', type=str, required=True)
    # args = argparse.parse_args()

    # main(args)

    copy_checkpoint_to_local()
