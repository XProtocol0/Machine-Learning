import numpy as np
import pandas as pd
import scipy.optimize as op
import plot
import costfunction
import gradient


data = pd.read_csv('Data/ex2data1.txt', header = None)
X = data.iloc[:,[0,1]]
y = data.iloc[:,2]
m=len(y)

print(data.head())

plot.plotdata(X,y)

(m, n) = X.shape
y = y[:, np.newaxis]
ones = np.ones((m,1))
X = np.hstack((ones, X))
#theta = np.zeros((X.shape[1],1))

#c = costfunction.cost(X, y, theta)
#print(c)

#test_theta = [[-24], [0.2], [0.2]]
#print(np.shape(test_theta))
#print(gradient.grad(X, y, test_theta))


#X = np.array([[1,2,3], [1,3,4]])
#y = np.array([[1],[0]])

#m , n = np.shape(X)
#initial_theta = np.zeros(n)
#initial_theta = initial_theta.reshape(n,1)
#print(np.shape(initial_theta))
#Result = op.minimize(fun = costfunction.cost, x0 = initial_theta, args =(X, y), method = 'TNC', jac = gradient.grad)
#optimal_theta = Result.x

#(m, n) = X.shape
#X = np.hstack((np.ones((m,1)), X))
#y = y[:, np.newaxis]
theta = np.zeros((n+1,1)) # intializing theta with all zeros
J = costfunction.cost(theta,X, y)
print(J)


temp = op.fmin_tnc(func = costfunction.cost, x0 = theta.flatten(), fprime = gradient.grad, args = (X, y.flatten()))
#the output of above function is a tuple whose first element #contains the optimized values of theta
theta_optimized = temp[0]
print(theta_optimized)