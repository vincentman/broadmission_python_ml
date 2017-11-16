#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 20:56:13 2017

@author: jerry
"""
#k-means
from sklearn.datasets import load_iris
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

iris = datasets.load_iris()

df_data = pd.DataFrame(iris['data'], columns=iris.feature_names)
df_target = pd.DataFrame(iris['target'], columns=['Target'])
df = pd.concat([df_data,df_target],axis=1)

km = KMeans(n_clusters=3)

km.fit(df_data)

labels = km.predict(df_data)
print(labels)

km.predict([[0.1,0.2,0.3,0.4],[5.9,3.0,5.1,1.8]])

###Visual
import matplotlib.pyplot as plt

#df_data.loc[:,['petal length (cm)','petal width (cm)']]

plt.scatter(df_data['petal length (cm)'],
                    df_data['petal width (cm)'],
                            c=labels)
plt.show()

###Evaluation
ks = range(1, 6)
inertias = []

for k in ks:

    km = KMeans(n_clusters=k)
    km.fit(df_data)
    inertias.append(km.inertia_)
    
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
