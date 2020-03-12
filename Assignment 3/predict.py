import numpy as np

def predict(all_theta, X):
    m = X.shape[0]
    X = np. hstack((np.ones((m,1)),X))

    pred = X @ all_theta.T

    return np.argmax(pred, axis = 1) + 1