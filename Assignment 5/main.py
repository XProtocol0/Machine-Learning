from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
from sklearn.preprocessing import StandardScaler
import linearRegCostFunction
import learningCurve
import polyFeatures
import gradientDes

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
theta = np.ones((2,1))
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

# Map X onto Polynomial features and normalize
p=8
X_poly = polyFeatures.polyFeatures(X, p)

sc_X=StandardScaler()
X_poly=sc_X.fit_transform(X_poly)
X_poly = np.hstack((np.ones((X_poly.shape[0],1)),X_poly))

# Map Xtest onto polynomial features and normalize
X_poly_test = polyFeatures.polyFeatures(Xtest, p)
X_poly_test = sc_X.transform(X_poly_test)
X_poly_test = np.hstack((np.ones((X_poly_test.shape[0],1)), X_poly_test))

# Map Xval onto polynomial features and normalize
X_poly_val = polyFeatures.polyFeatures(Xval, p)
X_poly_val = sc_X.transform(X_poly_val)
X_poly_val = np.hstack((np.ones((X_poly_val.shape[0],1)), X_poly_val))


theta_poly, J_history_poly = gradientDes.gradientDescent(np.zeros((9,1)),X_poly,y,0.3,20000,greek_lambda)

#theta2 = np.ones((9,1))
#temp = op.fmin_tnc(func = linearRegCostFunction.cost, x0 = theta2.flatten(), fprime = linearRegCostFunction.grad, args = (X_poly, y.flatten(), 0))
#theta_optimized2 = temp[0]
#theta_optimized2 = theta_optimized2[:, np.newaxis]


plt.scatter(X,y,marker="x",color="r")
plt.xlabel("Change in water level")
plt.ylabel("Water flowing out of the dam")
x_value=np.linspace(-55,65,2400)

# Map the X values and normalize
x_value_poly = polyFeatures.polyFeatures(x_value[:,np.newaxis], p)
x_value_poly = sc_X.transform(x_value_poly)
x_value_poly = np.hstack((np.ones((x_value_poly.shape[0],1)),x_value_poly))
y_value= x_value_poly @ theta_poly
plt.plot(x_value,y_value,"--",color="b")
plt.show()
