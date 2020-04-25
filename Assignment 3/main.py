from scipy.io import loadmat
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import costfunction
import gradient
import onevsall
import plot_j
import predict

data = loadmat('Data/ex3data1.mat')
X = data['X']
y = data['y']
#y[y== 10] = 0

#m, _ = np.shape(X)

#y = y.reshape(m, 1)

#print (y)

_, axis = plt.subplots(5,5,figsize=(5,5))
for i in range(5):
    for j in range(5):
       axis[i,j].imshow(X[np.random.randint(X.shape[0])].reshape((20,20), order = 'F'))          
       axis[i,j].axis('off') 
plt.show()


m = len(y)
ones = np.ones((m,1))
X = np.hstack((ones, X)) #add the intercep

'''
Now X = [[1 _ _ _ _ _ ....]
         [1 _ _ _ _ _ ....]
         [1 _ _ _ _ _ ....]
         ..
         ..
         ..
         [1 _ _ _ _ _ ....]]
with m x (n+1)

'''

#(m,n) = X.shape
print(np.shape(X))
#print(X)


greek_lambda = 0.1


temp_theta = np.array([-2, -1, 1, 2]).reshape(4,1)
temp_X = np.array([np.linspace(0.1, 1.5, 15)]).reshape(3,5).T
'''
w/0 T 
temp_X= [[0.1 0.2 0.3 0.4 0.5]
        [0.6 0.7 0.8 0.9 1. ]
        [1.1 1.2 1.3 1.4 1.5]]

'''
temp_X = np.hstack((np.ones((5,1)), temp_X))
temp_y = np.array([1, 0, 1, 0, 1]).reshape(5,1)
J = costfunction.cost(temp_theta, temp_X, temp_y, 3)
print(J)

grad = gradient.gradient(temp_theta, temp_X, temp_y, 3)
print(grad)

print("Running gradient descent...(estimated 5 seconds)")
greek_lambda = 0.1
num_labels = 10
all_theta, all_J = onevsall.onevall(X, y, num_labels, greek_lambda)

plot_j.plot(all_J)

print(np.shape(all_theta))


prediction = predict.predict(all_theta, X)
print(np.shape(prediction))
acc = sum(prediction[:, np.newaxis] == y)[0]*100/5000
print("Accuracy:", acc,"%")
