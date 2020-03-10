
import sigmoid
import numpy as np

def cost(theta, X, y, greek_lambda):
    m = len(y)
    J = (-1/m) * (y.transpose() @ np.log(sigmoid.sig(X @ theta)) + (1 - y.transpose()) @ np.log(1 - sigmoid.sig(X @ theta)))
#   print(theta.shape)
#   print(theta[1:].shape)
    reg = (greek_lambda/(2*m)) * (theta[1:].transpose() @ theta[1:])
    J = J + reg
    return J