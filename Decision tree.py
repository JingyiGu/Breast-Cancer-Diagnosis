#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 14:36:14 2018

@author: elaine
"""

###### Implement the decision tree classification method.
###### Run the decision tree on the breast cancer dataset using ten real-valued features only
###### (http://mlr.cs.umass.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29).
###### Apply a boosting method on the dataset and compare the results.

import numpy as np
    
def split(data,attr,value):     # split data into 2 parts
    new1 = data[np.nonzero(data[:,attr] > value)[0],:]
    new2 = data[np.nonzero(data[:,attr] <= value)[0],:]
    return new1, new2

def mainClass(data):
    classList = data[:,-1]
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), reverse=True)
    return sortedClassCount[0][0]

def entropy(data):          # calculate entropy
    numLabel = {}
    for sample in data:
        oriClass = sample[-1]
        if oriClass in numLabel.keys():
            numLabel[oriClass] += 1
        else:
            numLabel[oriClass] = 1
    
    entro = 0
    for key in numLabel:
        p = numLabel[key]/len(data)
        if p == 0:
            entro -= 0
        else:        
            entro -= p*np.log2(p)
    return entro

def selectSplits(data,leaf=mainClass, err=entropy):
    oriClass = data[:,-1]
    baseEntropy = entropy(data)
    bestInfoGain = 0.0; 
    bestFeature = 0
    bestValue = 0
    if len(set(oriClass)) == 1:
        return None,leaf(data)
    for i in range(np.shape(data)[1]-1):
        uniqueAttr = set(data[:,i])
        newEntropy = 0
        for attr in uniqueAttr:
            new1, new2 = split(data,i,attr)
            if (len(new1) < 15) or (len(new2) < 15): 
                continue
            p1 = len(new1)/len(data)
            p2 = len(new2)/len(data)
            newEntropy += p1*err(new1) + p2*err(new2)
            infoGain = baseEntropy - newEntropy
            if (infoGain > bestInfoGain):       #compare this to the best gain so far
                bestInfoGain = infoGain         #if better than current best, set to best
                bestFeature = i
                bestValue = attr
    new1, new2 = split(data,bestFeature,bestValue)
    if len(new1)<15 or len(new2)<15: 
        return None,leaf(data)
    return bestFeature, bestValue
    

def tree(data,leaf=mainClass, err=entropy):
    attr,value = selectSplits(data,leaf, err)
    if attr == None:
        return value,len(data)
    
    newtree = {}
    newtree['Index'] = attr
    newtree['Value'] = value
    new1, new2 = split(data,attr,value)
    newtree['left'] = tree(new1,leaf=mainClass, err=entropy)
    newtree['right'] = tree(new2,leaf=mainClass, err=entropy)
    return newtree



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

tree(data)

########### Boosting

from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(tree(data))
ada
