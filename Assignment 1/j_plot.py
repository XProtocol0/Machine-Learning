import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
import costfunction


def j_plot(X,y, theta):

    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    # initialize J_vals to a matrix of 0's
    J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

    # Fill out J_vals
    for i, theta0 in enumerate(theta0_vals):
        for j, theta1 in enumerate(theta1_vals):
            J_vals[i, j] = costfunction.cost(X, y, [theta0, theta1])
            
    # Because of the way meshgrids work in the surf command, we need to
    # transpose J_vals before calling surf, or else the axes will be flipped
    J_vals = J_vals.T

    # surface plot
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')
    plt.xlabel('theta0')
    plt.ylabel('theta1')
    plt.title('Surface')
    

    # contour plot
    # Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
   # breaks = np.logspace(-2, 3, 20)
    ax = plt.subplot(122)
    plt.contour(theta0_vals, theta1_vals, J_vals,linewidths=1, cmap='viridis' )  
    plt.xlabel('theta0')
    plt.ylabel('theta1')
    plt.plot(theta[0], theta[1], 'rx', ms=10, lw=2)
    plt.title('Contour, showing minimum')
    plt.show()
