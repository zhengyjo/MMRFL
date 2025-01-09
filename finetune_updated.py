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


import torch
from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)
import numpy as np

import chemprop.data.utils
from chemprop.data import set_cache_mol, empty_cache
from chemprop.features import get_available_features_generators

from chemprop.args import TrainArgs

args = chemprop.args.TrainArgs().parse_args()
mean_score, std_score = cross_validate_updated(args=args, train_func=run_training_updated)
