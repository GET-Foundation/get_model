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
import enformer_utils as utils


np.random.seed(42)
EXTENDED_SEQ_LENGTH = 393_216
SEQ_LENGTH = 196_608


def main(args):
    inputs = np.array(np.random.random((1, EXTENDED_SEQ_LENGTH, 4)), dtype=np.float32)
    inputs_cropped = enformer.TargetLengthCrop1D(SEQ_LENGTH)(inputs)


    checkpoint_gs_path = 'gs://dm-enformer/models/enformer/sonnet_weights/*'
    checkpoint_path = '/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/ckpts/sonnet'

    enformer_model = enformer.Enformer()
    checkpoint = tf.train.Checkpoint(module=enformer_model)
    latest = tf.train.latest_checkpoint(checkpoint_path)
    print(latest)
    status = checkpoint.restore(latest)

    # Using `is_training=False` to match TF-hub predict_on_batch function.
    restored_predictions = enformer_model(inputs_cropped, is_training=True)
    test = utils.get_metadata("human")
    human_dataset = utils.get_dataset('human', 'train').batch(1).repeat()
    breakpoint()


if __name__ == '__main__':  
    argparse = argparse.ArgumentParser()
    # argparse.add_argument('--organism', type=str, required=True)
    # argparse.add_argument('--subset', type=str, required=True)
    args = argparse.parse_args()

    main(args)
