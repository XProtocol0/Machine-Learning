import numpy as np
import costfunction
import gradient

def graddes(theta, X, y, alpha, num_iters, greek_lambda):
   # print(y)
    J_history = []

    for _ in range(num_iters):
        c = costfunction.cost(theta, X, y, greek_lambda)
        grad = gradient.gradient(theta, X, y, greek_lambda)
        theta = theta - (alpha * grad)
        J_history.append(c)
    return theta, J_history