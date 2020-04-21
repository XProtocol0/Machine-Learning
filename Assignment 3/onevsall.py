import numpy as np
import gradientdes


def onevall(X, y, num_labels, greek_lambda):
    (m, n) = X.shape
    initial_theta = np.zeros((n+1,1))
    all_theta = []
    all_J = []

    X = np.hstack((np.ones((m,1)), X)) # adding intercept term. 

    alpha = 1
    
    for i in range(1, num_labels+1):
        theta, J_history = gradientdes.graddes(initial_theta, X, np.where( y== i, 0.9, 0),alpha, 300, greek_lambda) 
        '''
         np.where( y== i, 0.9, 0)
         when y==i returns 0.9 otherwise 0


        '''
        all_theta.extend(theta)
        all_J.extend(J_history)
    
    return np.array(all_theta).reshape(num_labels, n + 1), all_J