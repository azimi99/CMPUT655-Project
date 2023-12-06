from collections import namedtuple
import json
import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42):
    """Set the seed for environment using torch.

    Args
    ----
    seed: int
        Random seed to set

    Returns
    -------
    None
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_config(path):
    with open(path, 'r') as f:
        configs = json.load(f)

    return configs
    

def hash_env(cur_pos, width):
    return cur_pos[0] * width + cur_pos[1]
