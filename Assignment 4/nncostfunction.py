import numpy as np
import sigmoidgradient
import sigmoid

def cost(nn_params, input_layer_size, hidden_layer_size, num_labels, X , y, greek_lambda): 
    theta1 = nn_params[:((input_layer_size+1)*hidden_layer_size)] # retrieving theta1 from nn_params. (input_layer_size+1)*hidden_layer_size) are taken from the starting.
    theta1 = theta1.reshape(hidden_layer_size, input_layer_size+1)

    theta2 = nn_params[((input_layer_size + 1)* hidden_layer_size):]  # Retrieving theta2 from nn_params.
    theta2 = theta2.reshape(num_labels, hidden_layer_size + 1)

    m = X.shape[0]
    J = 0 
    X = np.hstack((np.ones((m,1)), X))
    y10 = np.zeros((m, num_labels)) # its size is 5000x10

    a1 = sigmoid.sig( X @ theta1.T)
    a1 = np.hstack ((np.ones((m,1)), a1))
    a2 = sigmoid.sig(a1 @ theta2.T)


    for i in range(1,num_labels+1):
        y10[:,i-1][:,np.newaxis] = np.where(y==i,1,0)

    '''
        y10[:, i-1] gives 1-D array so [:,np.newaxis] is used to add another column.
        Above loop is used to change the value of elements of y10 array to 1, which should be the correct digit. 


    '''

    for j in range(num_labels):
        J = J + sum(-y10[:,j] * np.log(a2[:,j]) - (1-y10[:,j])*np.log(1-a2[:,j]))
    
    cost = 1/m* J
    reg_J = cost + greek_lambda/(2*m) * (np.sum(theta1[:,1:]**2) + np.sum(theta2[:,1:]**2)) # [:,1:] means whole array except 1st column(0th index); so 1 onwards. Here 1 is inclusive.
    
    #backpropagation to compute the gradients
    
    grad1 = np.zeros((theta1.shape))
    grad2 = np.zeros((theta2.shape))
    
    for i in range(m):
        xi= X[i,:] # 1 X 401, xi is 1-D array with size 401
        a1i = a1[i,:] # 1 X 26, a1i is 1-D array with size 26
        a2i = a2[i,:] # 1 X 10, a2i is 1-D array with size 10
        d2 = a2i - y10[i,:] # d2 is 1-D array with size 10
        d1 = theta2.T @ d2.T * sigmoidgradient.sigmoidgrad(np.hstack((1,xi @ theta1.T))) # (26x10 @ 10x1 = 26x1)* (1x401 @ 401x25 = 1x25 -> 1x26)  d1 = 1x26
        #print(np.shape(theta2.T @ d2.T))
        #print("Only this",np.shape(xi[:,np.newaxis].T))
        grad1 = grad1 + d1[1:][:,np.newaxis] @ xi[:,np.newaxis].T # 25x1 @ 1x401 = 25x401
        grad2 = grad2 + d2.T[:,np.newaxis] @ a1i[:,np.newaxis].T # 4x1 @ 1x26 = 4x26
        
    '''
    theta2.T @ d2.T return 1-D array of (26,) and np.hstack((1,xi @ theta1.T)) return 1-D array of (26,). * is used for element wise operation.
    so d1 is (26,)
    


    '''
    grad1 = 1/m * grad1
    grad2 = 1/m * grad2

    temp_theta1 = theta1
    temp_theta2 = theta2
    temp_theta1[:,0] = 0 # changing value of bais variable theta to 0. While calculating gradient. 
    temp_theta2[:,0] = 0 
    
    grad1_reg = grad1 + (greek_lambda/m) * temp_theta1
    grad2_reg = grad2 + (greek_lambda/m) * temp_theta2
    
    '''
    np.hstack((np.zeros((theta1.shape[0],1)),theta1[:,1:])
    
    
    print(type(cost))
    print(type(grad1))
    print(type(grad2))
    print(type(reg_J))
    print(type(grad1_reg))
    print(type(grad2_reg))
    '''
    return cost, grad1, grad2,reg_J, grad1_reg,grad2_reg