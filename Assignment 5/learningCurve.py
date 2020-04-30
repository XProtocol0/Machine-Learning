import numpy as np
import linearRegCostFunction
import scipy.optimize as op

def lc(X, y, Xval, yval, greek_lambda):
    """
    Returns the train and cross validation set errors for a learning curve
    """
    m=len(y)
    s = len(Xval)
    ones = np.ones((s,1))
    Xval1 = np.hstack((ones,Xval))
    n=X.shape[1]  
    err_train, err_val = [],[]
    
    for i in range(1,m+1):
        theta = np.zeros((n,1))
        print("The size of X is", np.shape(X[0:i, :]))
        print("The size of theta is", np.shape(theta))
        
        temp = op.fmin_tnc(func = linearRegCostFunction.cost, x0 = theta.flatten(), fprime = linearRegCostFunction.grad, args = (X[0:i, :], y[0:i,:].flatten(), 0))
        theta_optimized = temp[0]
        theta_optimized = theta_optimized[:, np.newaxis]
        #theta = gradientDescent(X[0:i,:],y[0:i,:],np.zeros((n,1)),0.001,3000,Lambda)[0]
        err_train.append(linearRegCostFunction.cost(theta_optimized, X[0:i,:], y[0:i,:],greek_lambda))
        err_val.append(linearRegCostFunction.cost(theta_optimized, Xval1, yval,greek_lambda))
        
    return err_train, err_val