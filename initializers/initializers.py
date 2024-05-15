import math

import numpy as np

# Shamelessly copied from https://stackoverflow.com/a/62249676
def he_init(shape):
    scale = 1/max(1., (2+2)/2.)
    limit = math.sqrt(3.0 * scale)

    weights = np.random.uniform(-limit, limit, size=shape)
    return weights

def random_init(shape):
    weights = np.random.uniform(-0.4, 0.4, size=shape)
    # weights = np.random.rand(shape[0], shape[1]) - 0.5
    return weights
