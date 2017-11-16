#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 17:54:11 2017

@author: jerry
"""

#k-fold CV
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()

df_data = pd.DataFrame(iris['data'], columns=iris.feature_names)
df_target = pd.DataFrame(iris['target'], columns=['Target'])
df = pd.concat([df_data,df_target],axis=1)

X_train, X_test, y_train, y_test = train_test_split(df_data, df_target,
                                                    test_size = 0.3,
                                                    random_state=42)

reg_all = linear_model.LinearRegression()
reg_all.fit(X_train, y_train)
cv_results = cross_val_score(reg_all, X_train, y_train, cv=5)

print(cv_results)
print("Average 5-Fold CV Score: {:.4f}".format(np.mean(cv_results)))

#########

