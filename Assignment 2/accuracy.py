import numpy as np
import sigmoid

def accuracy(X, y, theta):
    pred = [sigmoid.sig(np.dot(X, theta)) >= 0.5]
    acc = np.mean(pred == y)
    print(acc * 100)