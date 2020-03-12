from scipy.io import loadmat
import pandas as pd
import numpy as np
import sigmoid

data = loadmat('Data/ex3data1.mat')
X = data['X']
y = data['y']

mat2=loadmat("Data/ex3weights.mat")
Theta1=mat2["Theta1"] # Theta1 has size 25 x 401
Theta2=mat2["Theta2"] # Theta2 has size 10 x 26

m = X.shape[0]
X = np.hstack((np.ones((m,1)),X))
    
a1 = sigmoid.sig(X @ Theta1.T)
a1 = np.hstack((np.ones((m,1)), a1)) # hidden layer
a2 = sigmoid.sig(a1 @ Theta2.T) # output layer

pred2 = np.argmax(a2, axis = 1) + 1
#print(sum(pred2[:, np.newaxis]== y)[0])
#print(np.shape(pred2))
acc = sum(pred2[:,np.newaxis]==y)[0]/5000*100
print("Accuracy:", acc ,"%")

