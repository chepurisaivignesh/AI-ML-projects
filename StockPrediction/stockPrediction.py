# -*- coding: utf-8 -*-
#Stock prediction using SVM REGRESSION

import pandas as pd
import numpy as np

dataset = pd.read_csv('data.csv')

#Summarize Dataset

print(dataset.shape)
print(dataset.head(5))

#Segregation of Dataset into Input X & Output Y

X = dataset.iloc[:, :-1].values
X

Y = dataset.iloc[:, -1].values
Y

#Splitting Dataset for Testing our Model

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.20,random_state=0)

#Training Dataset using Support Vector Regression

from sklearn.svm import SVR
model = SVR()
model.fit(x_train,y_train)

ypred = model.predict(x_test)

from sklearn.metrics import r2_score,mean_squared_error
mse = mean_squared_error(y_test,ypred)
rmse=np.sqrt(mse)
print("Root Mean Square Error:",rmse)
r2score = r2_score(y_test,ypred)
print("R2Score",r2score*100)