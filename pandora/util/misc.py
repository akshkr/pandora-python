import numpy as np
import random
import os

__all__ = ['seed_everything']


def seed_everything(seed=34):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
