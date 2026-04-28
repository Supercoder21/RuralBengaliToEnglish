import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(T, d_model):
    PE = np.zeros((T, d_model))
    positions = np.arange(T)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angle_rates = positions /(10000 ** (2 * (dims // 2)/ d_model))
    PE[0:, 0::2] = np.sin(angle_rates[:, 0::2])
    PE[0:, 1::2] = np.cos(angle_rates[:, 1::2])
    return PE
