
import numpy as np

def sigmoidgrad(z):
    sigmoid = 1/(1 + np.exp(-z))
    return sigmoid * (1 - sigmoid)