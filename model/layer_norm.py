## NOTICE: Used to create PyTorch LayerNorm

import numpy as np
import math

def layer_norm(x, weights, bias, epsilon=10**-6):
    x_mean = np.mean(x)
    x_variance = np.var(x)
    x_norm = (x - x_mean)/np.sqrt(x_variance + epsilon)
    x_output = weights * x_norm + bias

    print("Mean:", x_mean)
    print("Variance:", x_variance)
    print("Normalized Values:", x_norm)

    return "Output: " + str(x_output)
