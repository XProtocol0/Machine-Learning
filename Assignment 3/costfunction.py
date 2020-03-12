import numpy as np
import sigmoid


def cost(theta, X, y, greek_lambda):
    m = len(y)
    h = sigmoid.sig(X @ theta)
    c = sum((y * np.log(h)) + ((1-y) * np.log(1-h)))/(-m)
    regularized_c= c + greek_lambda/(2*m)*sum(theta[1:]**2)
    #print(np.shape(regularized_c))
    return regularized_c
