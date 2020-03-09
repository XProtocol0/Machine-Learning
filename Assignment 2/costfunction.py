import numpy as np
import sigmoid 

def cost(theta, X, y):
    #m = len(y)
    #h= np.log(sigmoid.sig(np.dot(X, theta)))
    #print(np.shape(X))
    #print(np.shape(theta))
    #temp0 =  y*h
    #temp1 =  (1-y)*h
    #J = sum(temp0 + temp1)/(-m)
    #return J
    m = len(y)
    J = (-1/m) * np.sum(np.multiply(y, np.log(sigmoid.sig(X @ theta))) 
        + np.multiply((1-y), np.log(1 - sigmoid.sig(X @ theta))))
    return J

