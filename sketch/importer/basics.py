# General Imports
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import modin.pandas as mpd
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
