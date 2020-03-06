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

y = y[:, np.newaxis]
theta = np.zeros([3,1])
ones = np.ones((m,1))
X = np.hstack((ones, X))
print(X.shape)

c = costfunction.cost(X, y, theta, m)
print(c)




