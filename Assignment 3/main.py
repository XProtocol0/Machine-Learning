from scipy.io import loadmat
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import costfunction
import gradient


data = loadmat('Data/ex3data1.mat')
X = data['X']
y = data['y']

_, axarr = plt.subplots(5,5,figsize=(5,5))
for i in range(5):
    for j in range(5):
       axarr[i,j].imshow(X[np.random.randint(X.shape[0])].\
reshape((20,20), order = 'F'))          
       axarr[i,j].axis('off') 
plt.show()


m = len(y)
ones = np.ones((m,1))
X = np.hstack((ones, X)) #add the intercept
(m,n) = X.shape


greek_lambda = 0.1

theta = np.zeros((10,n)) #inital parameters
for i in range(10):
    digit_class = i if i else 10
    theta[i] = opt.fmin_cg(f = costfunction.cost, x0 = theta[i],  fprime = gradient.grad, args = (X, (y == digit_class).flatten(), greek_lambda), maxiter = 50)

pred = np.argmax(X @ theta.T, axis = 1)
pred = [e if e else 10 for e in pred]
print(np.mean(pred == y.flatten()) * 100)