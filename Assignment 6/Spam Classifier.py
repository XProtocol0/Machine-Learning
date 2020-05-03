
from scipy.io import loadmat
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import processEmail
import emailFeatures

file_contents = open("Text Files/emailSample1.txt","r").read()
vocabList = open("Text Files/vocab.txt","r").read()

vocabList=vocabList.split("\n")[:-1]


# Empty dictionary
vocabList_d={} 
for element_of_list in vocabList:
    value, key = element_of_list.split("\t")[:]
    vocabList_d[key] = value



print(file_contents)




word_indices= processEmail.processEmail(file_contents,vocabList_d)



features = emailFeatures.emailFeatures(word_indices,vocabList_d)
print("Length of feature vector: ", len(features))
print("Number of non-zero entries: ", np.sum(features))





spam_mat = loadmat("Data/spamTrain.mat")
X_train =spam_mat["X"]
y_train = spam_mat["y"]


C = 0.1
spam_svc = SVC(C=0.1,kernel ="linear")
spam_svc.fit(X_train,y_train.ravel())
print("Training Accuracy:",(spam_svc.score(X_train,y_train.ravel()))*100,"%")


spam_mat_test = loadmat("Data/spamTest.mat")
X_test = spam_mat_test["Xtest"]
y_test =spam_mat_test["ytest"]

spam_svc.predict(X_test)
print("Test Accuracy:",(spam_svc.score(X_test,y_test.ravel()))*100,"%")


weights = spam_svc.coef_[0]
weights_col = np.hstack((np.arange(1,1900).reshape(1899,1),weights.reshape(1899,1)))
df = pd.DataFrame(weights_col)

df.sort_values(by=[1],ascending = False,inplace=True)

predictors = []
idx=[]

for i in df[0][:15]:
    for keys, values in vocabList_d.items():
        if str(int(i)) == values:
            predictors.append(keys)
            idx.append(int(values))




print("Top predictors of spam:")

for i in range(15):
    print(predictors[i],"\t\t",round(df[1][idx[i]-1],6))
