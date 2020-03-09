import matplotlib.pyplot as plt
import numpy as np

def plotdata(X, y):
    admitted = y == 1
    p1 = plt.scatter(X[admitted][0].values, X[admitted][1].values, marker='+', color='k')
    p2 = plt.scatter(X[~admitted][0].values, X[~admitted][1].values, marker='o', color='#FBF047', edgecolors='k')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend((p1, p2), ('Admitted', 'Not admitted'))
    plt.show()

def plotboundary(X, y, theta_optimized):
    plot_x = [np.min(X[:,1]-2), np.max(X[:,2]+2)]
    plot_y = -1/theta_optimized[2]*(theta_optimized[0] + np.dot(theta_optimized[1],plot_x))  
    mask = y.flatten() == 1
    adm = plt.scatter(X[mask][:,1], X[mask][:,2], marker='+', color='k')
    not_adm = plt.scatter(X[~mask][:,1], X[~mask][:,2], marker='o', color='#FBF047', edgecolors='k')
    plt.plot(plot_x, plot_y)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend((adm, not_adm), ('Admitted', 'Not admitted'))
    plt.show()