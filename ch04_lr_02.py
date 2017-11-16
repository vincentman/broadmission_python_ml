# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:24:22 2017

@author: vincentkao
"""

################All Features
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

iris = datasets.load_iris()

df_data = pd.DataFrame(iris['data'], columns=iris.feature_names)
df_target = pd.DataFrame(iris['target'], columns=['Target'])
df = pd.concat([df_data,df_target],axis=1)

X_train, X_test, y_train, y_test = train_test_split(df_data, df_target,
                                                    test_size = 0.3,
                                                    random_state=42)

reg_all = linear_model.LinearRegression()
reg_all.fit(X_train, y_train)


y_pred = reg_all.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print('R^2: ', reg_all.score(X_test, y_test))
print('RMSE: ', rmse)