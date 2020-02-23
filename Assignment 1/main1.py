import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot
import costfunction
import gradientdes
from mpl_toolkits.mplot3d import Axes3D 
import j_plot

data = pd.read_csv('Data/ex1data1.txt', header = None) #read from dataset
X = data.iloc[:,0] # read first column
y = data.iloc[:,1] # read second column
m = len(y) # number of training example
print(data.head()) # view first few rows of the data

# plot.plotdate(X, y)

print(X.shape)
X = X[:,np.newaxis]
y = y[:,np.newaxis]
print(X.shape)
theta = np.zeros([2,1])
iterations = 1500
alpha = 0.01
ones = np.ones((m,1))
X = np.hstack((ones, X)) # adding the intercept term


J = costfunction.cost(X, y, theta)
print(J)

theta, J_history = gradientdes.gradientDescent(X, y, theta, alpha, iterations)
print(theta)

J = costfunction.cost(X, y, theta)
print(J)

plot.plotdata(X,y,theta)

predict1 = np.dot([1, 3.5], theta)
print('For population = 35,000, we predict a profit of')
print(predict1*10000)

predict2 = np.dot([1, 7], theta)
print('For population = 70,000, we predict a profit of')
print(predict2*10000)


j_plot.j_plot(X, y, theta)


