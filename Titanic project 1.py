# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 07:52:24 2018

@author: kanusaga
"""
import pandas as pd
from sklearn import tree
import io
import pydot
import os
titanic_train=pd.read_csv("C:\\Users\\kanusaga\\DataScience\\Titanic\\all\\train.csv")
titanic_train.shape
titanic_train.info()

#start with non categorical and non null columns
x_titanic_train = titanic_train[['Pclass', 'SibSp', 'Parch']] #X axis
y_titanic_train = titanic_train['Survived'] #Y axis

# now building decision tree model
dt = tree.DecisionTreeClassifier()
dt.fit(x_titanic_train, y_titanic_train)

# now predicting data using decision tree
titanic_test = pd.read_csv("C:\\Users\\kanusaga\\DataScience\\Titanic\\all\\test.csv")#reading test data
x_test = titanic_test[['Pclass', 'SibSp', 'Parch']]
#using predict

titanic_test['Survived'] = dt.predict(x_test) 
os.getcwd() #To get current working directory
titanic_test.to_csv("submission_Titanic.csv", columns=['PassengerId','Survived'], index=False)
