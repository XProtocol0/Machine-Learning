from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import linearRegCostFunction
import learningCurve

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
X1 = np.hstack((ones,X))
#print(X)

J = linearRegCostFunction.cost(theta, X1, y, 1)
grad = linearRegCostFunction.grad(theta, X1, y, 1)
 

print('Cost at theta = [1 ; 1]:', J)
print("Gradient at theta = [1 ; 1]:",grad)


greek_lambda = 0
temp = op.fmin_tnc(func = linearRegCostFunction.cost, x0 = theta.flatten(), fprime = linearRegCostFunction.grad, args = (X1, y.flatten(), 0))
theta_optimized = temp[0]
theta_optimized = theta_optimized[:, np.newaxis]


plt.scatter(X1[:,1:],y,marker="x",color="r")
plt.xlabel("Change in water level")
plt.ylabel("Water flowing out of the dam")
x_value=[x for x in range(-50,40)]
y_value=[y*theta_optimized[1]+theta_optimized[0] for y in x_value]
plt.plot(x_value,y_value,color="b")
plt.ylim(-5,40)
plt.xlim(-50,40)
plt.show()



#Xval_1 = np.hstack((np.ones((21,1)),Xval))
error_train, error_val = learningCurve.lc(X1, y, Xval, yval, greek_lambda)


plt.plot(range(12),error_train,label="Train")
plt.plot(range(12),error_val,label="Cross Validation",color="r")
plt.title("Learning Curve for Linear Regression")
plt.xlabel("Number of training examples")
plt.ylabel("Error")
plt.legend()
plt.show()

