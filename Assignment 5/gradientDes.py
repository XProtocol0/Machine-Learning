import numpy as np
import linearRegCostFunction

def gradientDescent(theta, X, y, alpha, num_iters, greek_lambda):
    """
    Take in numpy array X, y and theta and update theta by taking num_iters gradient steps
    with learning rate of alpha
    
    return theta and the list of the cost of theta during each iteration
    """
    
    
    J_history =[]
    
    for _ in range(num_iters):
        cost = linearRegCostFunction.cost(theta, X, y, greek_lambda)
        grad = linearRegCostFunction.cost(theta, X, y, greek_lambda)
        theta = theta - (alpha * grad)
        J_history.append(cost)
    
    return theta , J_history