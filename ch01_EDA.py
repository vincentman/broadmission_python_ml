#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 08:36:08 2017

@author: jerry
"""
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
iris = datasets.load_iris()

x = iris.data
y = iris.target
z= iris.target_names
df = pd.DataFrame(x, columns=iris.feature_names)




### Viewing Data
#df.dtypes
#df.describe()
#df.info()
#df.head(2)
#df.tail(1)
#df.T


##
#Renaming columns in pandas
#df.dtypes
#df.columns = ['sepal_length','sepal_width','petal_length','petal_width']
#df.sort_values(by='sepal_length', ascending=False)
#ascending=FALSE 從大排到小，反之從小排到大

#### Selection
#df['sepal_length'].head(9)
#df[0:3]
## Selection by Label
#df.loc[:,['sepal_length','sepal_width']].head(5)
## Selection by Position
#df.iloc[1:10,0:2] [列,行]

#Filling missing data
#df.dropna(how='any')
#df1.fillna(value=5)

### Merge
#xx = x + y
#xx_pd=pd.DataFrame(xx)
#pd.concat([xx_pd,xx_pd],axis=1) #行合併
#pd.concat([xx_pd,xx_pd],axis=0) #列合併

#df.drop(df.index[[0,1]])

pd.scatter_matrix(df, c = y, figsize=[8,8],
                      s=150, marker='D')