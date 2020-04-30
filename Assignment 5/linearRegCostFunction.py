import numpy as np


def cost (theta, X, y, greek_lambda):
    m = len(y)
   # ones = np.ones((m,1))
   # X = np.hstack((ones,X))
    h = X @ theta

    c = sum((h-y)**2)/(2*m)

    regularized_c = c + greek_lambda/(2*m)*sum(theta[1:]**2)
    
    return regularized_c

def grad (theta, X, y, greek_lambda):   
    m = len(y)
   # ones = np.ones((m,1))
   # X = np.hstack((ones,X))
    h = X @ theta
    
    grad1 = 1/m * X.T @ (h - y)
    grad2 = 1/m * X.T @ (h - y) + (greek_lambda/m * theta)
    grad = np.vstack((grad1[0], grad2[1:]))
    
    return grad