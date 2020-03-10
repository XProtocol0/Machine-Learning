import numpy as np
import sigmoid

def grad(theta, X, y, greek_lambda):
    m = len(y)
    temp = sigmoid.sig(np.dot(X, theta)) - y
    temp = np.dot(temp.T, X).T / m + theta * greek_lambda / m
    temp[0] = temp[0] - theta[0] * greek_lambda / m
    return temp

