# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 07:52:24 2018

@author: kanusaga
"""
import pandas as pd
from sklearn import tree
import io
import pydotplus #for using external files
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

#visualize the decission tree
objStringIO = io.StringIO() 
tree.export_graphviz(dt, out_file = objStringIO, feature_names = x_titanic_train.columns,filled=True,rounded=True)

# =============================================================================
os.environ["PATH"] += os.pathsep + 'C:/Users/kanusaga/AppData/Local/Continuum/anaconda3/pkgs/graphviz-2.38.0-4/Library/bin/graphviz/'
# #C:/Users/kanusaga/AppDataLocal/Continuum/anaconda3/pkgs/graphviz-2.38-hfd603c8_2/Library/bin/
# =============================================================================

#Use out_file = objStringIO to getvalues()
file1 = pydotplus.graph_from_dot_data(objStringIO.getvalue()) #[0]
type(file1)
#os.chdir("C:\\Users\\kanusaga\\DataScience\\Titanic\\all\\")
file1.write_pdf("DecissionTree1.pdf")
os.getcwd()



titanic_test['Survived'] = dt.predict(x_test) 
os.getcwd() #To get current working directory
titanic_test.to_csv("submission_Titanic.csv", columns=['PassengerId','Survived'], index=False)
