import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import featureNormalize
import costfunction
import gradientdes

data = pd.read_csv('Data/ex1data2.txt', header = None) #read from dataset
x = data.iloc[:, [0,1]] # read first column
y = data.iloc[:,2] # read second column
m = len(y) # number of training example


X = featureNormalize.fn(x,y)

ones = np.ones((m,1))
X = np.hstack((ones, X))
y = y[:,np.newaxis]
theta = np.zeros([3,1])
iterations = 500
alpha = 0.01


J = costfunction.cost(X, y, theta)
print(J)


theta, J_history = gradientdes.gradientDescent(X, y, theta, alpha, iterations)
print(theta)

J = costfunction.cost(X, y, theta)
print(J)

plt.plot(J_history)
plt.xlabel("Iterations")
plt.ylabel("Theta")
plt.title("Cost function using gradient descent")
plt.show()
