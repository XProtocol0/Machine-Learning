import numpy as np

def fn(x,y):
    meanx = np.mean(x, axis = 0)
    stdx = np.std(x, axis = 0)
   # meany = np.mean(y, axis= 0)
    #stdy = np.std(y, axis = 0)
    
    X = (x- meanx)/stdx
    #Y = (y- meany)/stdy
    
    return X