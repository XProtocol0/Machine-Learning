import numpy as np

def cost(X, y, theta):
    m=len(y)
    temp = np.dot(X, theta) - y
    return np.sum(np.power(temp, 2)) / (2*m)


