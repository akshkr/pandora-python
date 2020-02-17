# General Imports
import matplotlib.pyplot as plt
from time import time
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import random
import os


# Seed to make every process deterministic
def seed_everything(seed=27):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
