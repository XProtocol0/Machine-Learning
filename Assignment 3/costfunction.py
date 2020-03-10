import numpy as np
import sigmoid


def cost(theta, X, y, greek_lambda):
    m = len(y)
    temp1 = np.multiply(y, np.log(sigmoid.sig(np.dot(X, theta))))
    temp2 = np.multiply(1-y, np.log(1-sigmoid.sig(np.dot(X, theta))))
    return np.sum(temp1 + temp2) / (-m) + np.sum(theta[1:]**2) * greek_lambda / (2*m)