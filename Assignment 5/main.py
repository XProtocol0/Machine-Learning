from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import linearRegCostFunction


data = loadmat('Data/ex5data1.mat')

X = data['X']
y = data['y']
Xval = data['Xval']
yval = data['yval']
Xtest = data['Xtest']
ytest = data['ytest']

plt.scatter(X,y, marker='x', c ='r')
plt.xlabel("Change in water level")
plt.ylabel("Water flowing out of the dam")
plt.show()



m = len(y)
theta =np.ones((2,1))
print(np.shape(theta))
ones = np.ones((m,1))
X = np.hstack((ones,X))
print(X)

J = linearRegCostFunction.cost(theta, X, y, 1)
grad = linearRegCostFunction.grad(theta, X, y, 1)
 

print('Cost at theta = [1 ; 1]:', J)
print("Gradient at theta = [1 ; 1]:",grad)
#         '\n(this value should be about 303.993192)\n'], J);

greek_lambda = 0
temp = op.fmin_tnc(func = linearRegCostFunction.cost, x0 = theta.flatten(), fprime = linearRegCostFunction.grad, args = (X, y.flatten(), 0))
theta_optimized = temp[0]
theta_optimized = theta_optimized[:, np.newaxis]
#print(np.shape(theta_optimized))


plt.scatter(X[:,1],y,marker="x",color="r")
plt.xlabel("Change in water level")
plt.ylabel("Water flowing out of the dam")
x_value=[x for x in range(-50,40)]
y_value=[y*theta_optimized[1]+theta_optimized[0] for y in x_value]
plt.plot(x_value,y_value,color="b")
plt.ylim(-5,40)
plt.xlim(-50,40)
plt.show()