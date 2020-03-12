import numpy as np
import sigmoid

def gradient(theta, X, y, greek_lambda):
    m = len(y)
    h = sigmoid.sig(X @ theta)
    j0 = (1/m)* (X.T @ (h - y))[0]
    j1= (1/m)* ((X.T @ (h - y))[1:] + greek_lambda*theta[1:])
    grad = np.vstack((j0[:, np.newaxis], j1))
    return grad