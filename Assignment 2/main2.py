import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt  
import mapfeature 
import sigmoid
import regularizedgradient
import regularlizedcostfuntion


data = pd.read_csv('Data/ex2data2.txt', header = None)
X = data.iloc[:,[0,1]]
y = data.iloc[:,2]
data.head()

admitted = y == 1
passed = plt.scatter(X[admitted][0].values, X[admitted][1].values,  marker='+', color='k')
failed = plt.scatter(X[~admitted][0].values, X[~admitted][1].values, marker='o', color='#FBF047', edgecolors='k')
plt.xlabel('Microchip Test1')
plt.ylabel('Microchip Test2')
plt.legend((passed, failed), ('Passed', 'Failed'))
plt.show()


X = mapfeature.mapFeature(X)
print (np.shape(X))

(m, n) = X.shape
y = y[:, np.newaxis]
theta = np.zeros((n,1))

greek_lambda = 1
J = regularlizedcostfuntion.cost(theta, X, y, greek_lambda)
print(J)

output = opt.fmin_tnc(func = regularlizedcostfuntion.cost, x0 = theta.flatten(), fprime = regularizedgradient.grad, args = (X, y.flatten(), greek_lambda))
theta = output[0]
print(theta) # theta contains the optimized values

#print(np.shape(sigmoid.sig(X @ theta)))
pred = [sigmoid.sig(X @ theta) >= 0.5]
#print(np.shape(pred))
print(np.mean(pred == y.flatten()) * 100)



a = np.linspace(-1, 1.5, 100)
b = np.linspace(-1, 1.5, 100)
z = np.zeros((len(a), len(b)))

for i in range(len(a)):
    for j in range(len(b)):
        z[i,j] = np.dot(mapfeature.feature_plotting(a[i], b[j]), theta)

#admitted = y.flatten() == 1
X = data.iloc[:,:-1]
passed = plt.scatter(X[admitted][0].values, X[admitted][1].values,  marker='+', color='k')
failed = plt.scatter(X[~admitted][0].values, X[~admitted][1].values, marker='o', color='#FBF047', edgecolors='k')
plt.contour(a,b,z,0)
plt.xlabel('Microchip Test1')
plt.ylabel('Microchip Test2')
plt.legend((passed, failed), ('Passed', 'Failed'))
plt.show()