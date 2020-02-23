import matplotlib.pyplot as plt
import numpy as np


def plotdata(X,y, theta):

    plt.scatter(X[:,1], y, marker='x', color='r')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.plot(X[:,1], np.dot(X, theta))
    plt.show()
