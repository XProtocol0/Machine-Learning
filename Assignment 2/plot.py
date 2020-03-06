import matplotlib.pyplot as plt

def plotdata(X, y):
    admitted = y == 1
    p1 = plt.scatter(X[admitted][0].values, X[admitted][1].values, marker='+', color='k')
    p2 = plt.scatter(X[~admitted][0].values, X[~admitted][1].values, marker='o', color='#FBF047', edgecolors='k')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend((p1, p2), ('Admitted', 'Not admitted'))
    plt.show()

