from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
from sklearn.svm import SVC

data = loadmat('Data/ex6data1.mat')

X = data['X']
y = data['y']


m, n = np.shape(X)

positive = (y==1).reshape(m,1)
negative = (y==0).reshape(m,1)

'''
X[positive[:,0],0]  gives value for which y is 1 in the 1st column of X
'''

plt.scatter(X[positive[:,0],0],X[positive[:,0],1], c='#000000', marker='+')
plt.scatter(X[negative[:,0],0],X[negative[:,0],1], c='#f5f242', marker='o', edgecolors='#000000')
plt.show()

classifier = SVC(kernel="linear")
classifier.fit(X,np.ravel(y))

plt.figure(figsize=(8,6))
plt.scatter(X[positive[:,0],0], X[positive[:,0],1], c='#000000', marker='+')
plt.scatter(X[negative[:,0],0], X[negative[:,0],1], c='#f5f242', marker='o', edgecolors='#000000')

# plotting the decision boundary
X_1, X_2 = np.meshgrid(np.linspace(X[:,0].min(), X[:,0].max(), 100), np.linspace(X[:,1].min(), X[:,1].max(), 100))
plt.contour(X_1,X_2,classifier.predict(np.array([X_1.ravel(),X_2.ravel()]).T).reshape(X_1.shape),1, colors="b")
plt.xlim(0,4.5)
plt.ylim(1.5,5)
plt.show()


classifier2 = SVC(C=100,kernel="linear")
classifier2.fit(X,np.ravel(y))

plt.figure(figsize=(8,6))
plt.scatter(X[positive[:,0],0], X[positive[:,0],1], c='#000000', marker='+')
plt.scatter(X[negative[:,0],0], X[negative[:,0],1], c='#f5f242', marker='o', edgecolors='#000000')

# plotting the decision boundary
X_3,X_4 = np.meshgrid(np.linspace(X[:,0].min(), X[:,0].max(), 100), np.linspace(X[:,1].min(), X[:,1].max(), 100))
plt.contour(X_3,X_4,classifier2.predict(np.array([X_3.ravel(),X_4.ravel()]).T).reshape(X_3.shape),1,colors="b")
plt.xlim(0,4.5)
plt.ylim(1.5,5)
plt.show()