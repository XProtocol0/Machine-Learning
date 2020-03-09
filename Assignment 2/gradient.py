import numpy as np
import sigmoid 

def grad(theta, X, y):
    m = len(y)
    temp = sigmoid.sig(np.dot(X, theta)) - y
    print(np.shape(temp))
    return np.dot(X.transpose(), temp)/m
    #return ((1/m) * X.T @ (sigmoid.sig(X @ theta) - y))