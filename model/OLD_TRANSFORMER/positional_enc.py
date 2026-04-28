# Note: positional_encoding() is directly used in transformer.py inside the
# PositionalEncoding class. The output is computed once at initialisation and stored as a buffer, then added to token embeddings in the forward pass.
import numpy as np
def positional_encoding(T, d_model):
    PE = np.zeros((T, d_model)) # positional encoding matrix
    positions=np.arange(T)[:,np.newaxis]
    dims=np.arange(d_model)[np.newaxis, :]
    angle_rates=positions /(10000**(2*(dims // 2)/d_model))
    PE[0:, 0::2] = np.sin(angle_rates[:, 0::2]) # if even dimension
    PE[0:, 1::2] = np.cos(angle_rates[:,1::2]) # if odd dimension
    return PE
