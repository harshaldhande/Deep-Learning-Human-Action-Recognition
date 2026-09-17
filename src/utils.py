import os
import random
import numpy as np
import tensorflow as tf

def set_seed(seed=27):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

def ensure_directory(path):
    os.makedirs(path, exist_ok=True)
