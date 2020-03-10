import numpy as np
import sigmoid

def grad(theta, X, y, greek_lambda):
    m = len(y)
    grad = np.zeros([m,1])
    grad = (1/m) * X.T @ (sigmoid.sig(X @ theta) - y)
    grad[1:] = grad[1:] + (greek_lambda / m) * theta[1:]
    return grad

