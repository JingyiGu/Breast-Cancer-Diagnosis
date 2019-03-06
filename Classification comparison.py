#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 19:39:14 2018

@author: elaine
"""

###### –Use 5-fold cross-validation on the breast cancer dataset 
###### using ten real-valued features only to compare the performance for all these methods: 
###### (1) decision tree, (2) random forest, 
###### (3) gradient tree boosting, (4) SVM, (5) LDA, 
###### (6) logistic regression, (7) KNN, (8) AdaBoost,
###### and (9) XGBoost (XGBoost is optional). 
###### You may also compare the classification methods using the ROC curve.

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

bc = open("wdbc.data.txt","r")
data = []
for line in bc:
    sample = []
    for value in line.split(',')[2:12]:
        sample.append(float(value))
    if line.split(',')[1] == 'M':
        c = 1
    else: c = 0
    sample.append(c)
    data.append(sample)   
    
labels = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 
          'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']

data = np.asarray(data)


x = data[:,:10]
y = data[:,-1]

def performModel(x,y,model,name):
    kf = KFold(n_splits=5)
    a = []
    for train,test in kf.split(x):
        x_train, x_test = x[train], x[test]
        y_train, y_test = y[train], y[test]
        if name == "XGBoost":
            score = model.fit(x_train,y_train).predict(x_test)
        else:
            score = model.fit(x_train,y_train).predict_proba(x_test)[:,1]
        fpr, tpr, thresholds = roc_curve(y_test, score)
        roc_auc = auc(fpr,tpr)
        a.append(roc_auc)
        plt.plot(fpr,tpr,marker = 'o',label = roc_auc)

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.title(name + ' ROC curve')
    plt.show()
    print('Average AUC: ' + str(np.mean(a)))

performModel(x,y,DecisionTreeClassifier(),"Decision Tree")
performModel(x,y,RandomForestClassifier(),"Random Forest")
performModel(x,y,GradientBoostingClassifier(),"Gradient Tree Boosting")
performModel(x,y,SVC(kernel='linear', probability=True),"SVM")
performModel(x,y,LinearDiscriminantAnalysis(),"LDA")
performModel(x,y,LogisticRegression(),"Logistic Regression")
performModel(x,y,KNeighborsClassifier(),"KNN")
performModel(x,y,AdaBoostClassifier(DecisionTreeClassifier()),"AdaBoost")
performModel(x,y,xgb.XGBRegressor(objective ='reg:linear'),"XGBoost")


