import numpy as np
import costfunction

def gradientDescent(X, y, theta, alpha, iterations):
    m=len(y)
    J_history=[]
    for _ in range(iterations):
        jtheta = np.dot(X, theta) - y
        jtheta = np.dot(X.transpose(), jtheta)
        theta = theta - (alpha/m) * jtheta
        J_history.append(costfunction.cost(X,y,theta))
    return theta, J_history

