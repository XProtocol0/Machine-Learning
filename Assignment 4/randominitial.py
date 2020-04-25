import numpy as np

def randInitializeWeights(L_in, L_out):
    """
    randomly initializes the weights of a layer with L_in incoming connections and L_out outgoing connections.
    """
    
    e = 0.12
    
    W = np.random.rand(L_out,L_in +1) *(2*e) -e
    
    return W