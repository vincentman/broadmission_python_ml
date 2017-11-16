#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 11:35:42 2017

@author: jerry
"""

from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

iris = datasets.load_iris()

df_data = pd.DataFrame(iris['data'], columns=iris.feature_names)
df_target = pd.DataFrame(iris['target'], columns=['Target'])
df = pd.concat([df_data,df_target],axis=1)


reg = linear_model.LinearRegression()

iris_pl_re = df['petal length (cm)'].values.reshape(-1, 1)
iris_pw_re = df['petal width (cm)'].values.reshape(-1, 1)

reg.fit(iris_pl_re, iris_pw_re)
score = reg.score(iris_pl_re, iris_pw_re)
print("score: ", score)

prediction_space = np.linspace(min(iris_pl_re),
                               max(iris_pl_re)).reshape(-1, 1)

plt.scatter(iris_pl_re, iris_pw_re, color = 'blue')
plt.plot(prediction_space, reg.predict(prediction_space),
         color='black', linewidth=2)

plt.show()


