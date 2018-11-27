# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 07:55:38 2018

@author: kanusaga
"""

import pandas as pd
from sklearn import tree
#from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import graphviz

titanic_train = pd.read_csv("C:\\Users\\kanusaga\\DataScience\\Titanic\\all\\train.csv")
print(type(titanic_train))


#EDA
titanic_train.shape
titanic_train.info()

#applying one hot encoding

titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass','Sex','Embarked'])
titanic_train1.shape
titanic_train1.info()


X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], 1)

y_train = titanic_train['Survived']

#section 1A

dt1 = tree.DecisionTreeClassifier()
dt1.fit(X_train,y_train)

#Apllying K-fold technique and finding out cross validation score.

cv_score1 = cross_val_score(dt1, X_train, y_train, cv=10)

print(cv_score1)
print(cv_score1.mean())

#section 1B

print(dt1.score(X_train,y_train))


#section 2A
#tune model manually by passing different arguments to decision tree

dt2 = tree.DecisionTreeClassifier(max_depth=4)
dt2.fit(X_train,y_train)

cv_score2 = cross_val_score(dt2, X_train, y_train, cv=10)
print(cv_score2)
print(cv_score2.mean())

#section 2B

print(dt2.score(X_train,y_train))

#automate model tuning process using grid search method

dt3 = tree.DecisionTreeClassifier()
param_grid = {'max_depth': [5,8,10],'min_samples_split': [2,4,5],'criterion': ['gini','entropy']}
print(param_grid)

dt3_grid = GridSearchCV(dt3, param_grid, cv=10, n_jobs=5)
dt3_grid.fit(X_train,y_train)

print(dt3_grid.grid_scores_)

final_model = dt3_grid.best_estimator_
print(dt3_grid.best_score_)

print(dt3_grid.score(X_train,y_train))


graph_data = tree.export_graphviz(final_model, out_file=None, 
                         feature_names=X_train.columns,  
                         #class_names=x_titanic_train.,  
                         filled=True, rounded=True,  
                         special_characters=True) 

graph = graphviz.Source(graph_data)

type(graph)

graph.render('Tuned_DecissionTree')