import numpy as np
import sigmoid 

def grad(X, y, theta, m):
    temp = sigmoid.sig(np.dot(X, theta)) - y
    return np.dot(X.transpose(), temp)/m