import numpy as np
import sigmoid 

def cost(X, y , theta, m):
    h= np.log(sigmoid.sig(np.dot(X, theta)))
    temp0 =  y*h
    temp1 =  (1-y)*h
    return sum(temp0 + temp1)/(-m)


