#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 08:59:05 2017

@author: jerry
"""

from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris.data.shape

df_data = pd.DataFrame(iris['data'], columns=iris.feature_names)
df_target = pd.DataFrame(iris['target'], columns=['Target'])
df = pd.concat([df_data,df_target],axis=1)

#Create a KNN Classifier
knn = KNeighborsClassifier(n_neighbors=6)
#Training
knn.fit(df.iloc[0:150,0:4], df['Target'])
acc = knn.score(df.iloc[0:150,0:4], df['Target'])
print("accuracy: ", acc)

x_new = np.array([[5.9, 3.0, 5.1, 1.8], [0.1, 0.2, 0.3, 0.4]])
df_new = df_new=df.iloc[0:3,0:4]
#Single Row
knn.predict([5.9, 3.0, 5.1, 1.8])
#Array
knn.predict(x_new)
print('Prediction x_new:',knn.predict(x_new))
print('Predict proba: \n', knn.predict_proba(x_new))
#Pandas
knn.predict(df_new)
knn.predict(df.iloc[145:149,0:4])
#Print
print('Predict df:',knn.predict(df.iloc[145:149,0:4]))

