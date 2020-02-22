import matplotlib.pyplot as plt

def plotdate(X,y):

    plt.scatter(X, y, color='b', marker='x')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()
