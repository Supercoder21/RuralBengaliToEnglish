# Note: layer_norm() was integrated into transformer.py via the LayerNorm class.
# PyTorch's F.layer_norm is used in the forward pass for autograd compatibility, but the logic here is preserved as the original reference implementation.
import numpy as np
import math

def layer_norm(x, weights, bias, epsilon=10**-6):
    x_mean = np.mean(x)
    x_variance = np.var(x)
    x_norm = (x - x_mean)/np.sqrt(x_variance + epsilon) #epsilon used to stop DivisionByZero errors
    x_output = weights * x_norm + bias
    return x_output    
