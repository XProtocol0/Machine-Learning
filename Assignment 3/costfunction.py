import numpy as np
import sigmoid


def cost(theta, X, y, greek_lambda):
    m = len(y)
    h = sigmoid.sig(X @ theta) # This is vector multiplication. 
    '''
    h = [[ _ ]    each row is for 1 training example
         [ _ ]
         [ _ ]
         [ _ ]
         [ _ ]]

    '''
    c = sum((y * np.log(h)) + ((1-y) * np.log(1-h)))/(-m) 
    '''
    np.log, y * np.log(h)) are element wise and not vector operation.
    sum() adds all the elements in the matrix.
    '''


    regularized_c= c + greek_lambda/(2*m)*sum(theta[1:]**2)
    #print(np.shape(regularized_c))
    return regularized_c
