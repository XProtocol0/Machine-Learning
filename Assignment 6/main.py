from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
from sklearn.svm import SVC
import dataset3Params

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

classifier = SVC(kernel='linear')
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


classifier2 = SVC(C=100,kernel='linear')
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




# SVM with gaussian kernels

mat2 = loadmat('Data/ex6data2.mat')
X2 = mat2["X"]
y2 = mat2["y"]



m2,n2 = X2.shape[0],X2.shape[1]
positive2 = (y2==1).reshape(m2,1)
negative2 = (y2==0).reshape(m2,1)



plt.figure(figsize=(8,6))
plt.scatter(X2[positive2[:,0],0],X2[positive2[:,0],1],c='#000000',marker='+')
plt.scatter(X2[negative2[:,0],0],X2[negative2[:,0],1],c='#f5f242', marker='o', edgecolors='#000000')
plt.xlim(0,1)
plt.ylim(0.4,1)
plt.show()



classifier3 = SVC(kernel='rbf',gamma=30)
classifier3.fit(X2,y2.ravel())



plt.figure(figsize=(8,6))
plt.scatter(X2[positive2[:,0],0],X2[positive2[:,0],1],c='#000000',marker='+')
plt.scatter(X2[negative2[:,0],0],X2[negative2[:,0],1], c='#f5f242', marker='o', edgecolors='#000000')



# plotting the decision boundary
X_5,X_6 = np.meshgrid(np.linspace(X2[:,0].min(), X2[:,0].max(), 100),np.linspace(X2[:,1].min(), X2[:,1].max(), 100))
plt.contour(X_5,X_6,classifier3.predict(np.array([X_5.ravel(),X_6.ravel()]).T).reshape(X_5.shape),1,colors="b")
plt.xlim(0,1)
plt.ylim(0.4,1)
plt.show()



# 3rd DataSet


mat3 = loadmat('Data/ex6data3.mat')
X3 = mat3["X"]
y3 = mat3["y"]
Xval = mat3["Xval"]
yval = mat3["yval"]

m3,n3 = X3.shape[0],X3.shape[1]
pos3,neg3= (y3==1).reshape(m3,1), (y3==0).reshape(m3,1)
plt.figure(figsize=(8,6))
plt.scatter(X3[pos3[:,0],0],X3[pos3[:,0],1], c='#000000', marker='+')
plt.scatter(X3[neg3[:,0],0],X3[neg3[:,0],1], c='#f5f242', marker='o', edgecolors='#000000')

values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
C, gamma = dataset3Params.dataset3Params(X3, y3.ravel(), Xval, yval.ravel(),values)
classifier4 = SVC(C=C, gamma=gamma)
classifier4.fit(X3,y3.ravel())



plt.figure(figsize=(8,6))
plt.scatter(X3[pos3[:,0],0],X3[pos3[:,0],1], c='#000000', marker="+")
plt.scatter(X3[neg3[:,0],0],X3[neg3[:,0],1], c='#f5f242', marker='o', edgecolors='#000000')

# plotting the decision boundary
X_7,X_8 = np.meshgrid(np.linspace(X3[:,0].min(),X3[:,0].max(),num=100),np.linspace(X3[:,1].min(),X3[:,1].max(),num=100))
plt.contour(X_7,X_8,classifier4.predict(np.array([X_7.ravel(),X_8.ravel()]).T).reshape(X_7.shape),1,colors="b")
plt.xlim(-0.6,0.3)
plt.ylim(-0.7,0.5)
plt.show()