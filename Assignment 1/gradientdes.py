import numpy as np
import costfunction

def gradientDescent(X, y, theta, alpha, iterations):
    m=len(y)
    J_history=[]
    for _ in range(iterations):
        temp = np.dot(X, theta) - y
        temp = np.dot(X.transpose(), temp)
        theta = theta - (alpha/m) * temp
        J_history.append(costfunction.cost(X,y,theta))
    return theta, J_history

