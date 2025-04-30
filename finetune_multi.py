import json
import os
from tempfile import TemporaryDirectory
import pickle
from typing import List, Optional
from typing_extensions import Literal
from packaging import version
from warnings import warn
from cross_validate_updated import cross_validate_updated
from chemprop.train.run_training_updated import run_training_updated
import argparse


import torch
from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)
import numpy as np

import chemprop.data.utils
from chemprop.data import set_cache_mol, empty_cache
from chemprop.features import get_available_features_generators

from chemprop.args import TrainArgs
import ast

# Create the parser
parser = argparse.ArgumentParser(description="A simple argument parser example")

# Add arguments
parser.add_argument('--dir', type=str,default='data/bace.csv', help="data path")
parser.add_argument('--dataset_type',type=str, default='classification', help="dataset type")
parser.add_argument('--gpu',type=str, default='0', help="gpu device")
parser.add_argument('--encoders',type=str, default="['./pretrained_weight/smiles.pt','./pretrained_weight/image.pt','./pretrained_weight/nmr.pt','./pretrained_weight/fingerprint.pt','./pretrained_weight/peak.pt']", help="encoder directory")
parser.add_argument('--seed',type=str, default='42', help="seed number")

args = parser.parse_args()


# encoder_paths_arr = ['M3_KMGCL_encoder_smiles_alpha_0.0_01102024.pt','M3_KMGCL_encoder_image_alpha_0.0_01102024.pt','M3_KMGCL_encoder_nmr_alpha_0.0_01102024.pt','M3_KMGCL_encoder_fusion_fingerprint_alpha_0.0_01102024.pt','M3_KMGCL_encoder_fusion_nmr_alpha_1_01102024.pt'] 
# encoder_paths = ','.join(str(v) for v in encoder_paths_arr)

encoder_paths_arr = ast.literal_eval(args.encoders) 

data_dir = args.dir
data_dir_arr = data_dir.split("/")
dataset_type = args.dataset_type
gpu = args.gpu
seed = args.seed

arguments = [
    '--data_path', data_dir,
    '--dataset_type', dataset_type,
    '--multi_modality_ensemble','True',
    '--save_dir', '{}_test_checkpoints_multi_Late'.format(data_dir_arr[-1][:-4]),
    '--epochs', '50',
    '--encoder_path',encoder_paths_arr,
    '--batch_size', '256', 
    '--split_type','scaffold_balanced',
    '--num_folds','1',
    '--init_lr','1e-4',
    '--seed',seed,
    '--gpu',gpu,
    '--save_smiles_splits'
]

args = TrainArgs().parse_args(arguments)

mean_score, std_score = cross_validate_updated(args=args, train_func=run_training_updated)
