
from scipy.io import loadmat
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import nncostfunction
import randominitial
import nngradientdescent
import predict

data = loadmat('Data/ex4data1.mat')
X = data['X']
y = data['y']




_, axis = plt.subplots(5,5,figsize=(5,5))
for i in range(5):
    for j in range(5):
       axis[i,j].imshow(X[np.random.randint(X.shape[0])].reshape((20,20), order = 'F'))          
       axis[i,j].axis('off') 
plt.show()


mat2 = loadmat("Data/ex4weights.mat")
theta1 = mat2["Theta1"]  # Theta1 has size 25 x 401. As there are 400 pixels so 400 input and 1st hindden layer has size 25 units.
theta2 = mat2["Theta2"]  # Theta2 has size 10 x 26. As hidden layer has 25 units and output layer have 10 units.

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
nn_params = np.append(theta1.flatten(), theta2.flatten()) # tota1.flatten() has shape (10025, ), theta2.flatten() has shape (260,) and appends them to form (10285, ).
J,reg_J = nncostfunction.cost(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, 1)[0:4:3]   #returns 1st(0th index) and 4th(3rd index) return variables of function.   
print("Cost at parameters (non-regularized):",J,"\nCost at parameters (Regularized):",reg_J)


initial_Theta1 = randominitial.randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randominitial.randInitializeWeights(hidden_layer_size, num_labels)
initial_nn_params = np.append(initial_Theta1.flatten(),initial_Theta2.flatten())

debug_J  = nncostfunction.cost(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, 3)
print("Cost at (fixed) debugging parameters (w/ lambda = 3):",debug_J[3])



nnTheta, nnJ_history = nngradientdescent.grad(X,y,initial_nn_params,0.8,800,1,input_layer_size, hidden_layer_size, num_labels)
Theta1 = nnTheta[:((input_layer_size+1) * hidden_layer_size)].reshape(hidden_layer_size,input_layer_size+1)
Theta2 = nnTheta[((input_layer_size +1)* hidden_layer_size ):].reshape(num_labels,hidden_layer_size+1)

pred3 = predict.predict(Theta1, Theta2, X)
print("Training Set Accuracy:",sum(pred3[:,np.newaxis]==y)[0]/5000*100,"%")

