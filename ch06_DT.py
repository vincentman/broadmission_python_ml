#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 18:45:51 2017

@author: jerry
"""

#DT
from sklearn import datasets
from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


iris = datasets.load_iris()

df_data = pd.DataFrame(iris['data'], columns=iris.feature_names)
df_target = pd.DataFrame(iris['target'], columns=['Target'])
df = pd.concat([df_data,df_target],axis=1)

dtc = tree.DecisionTreeClassifier()
#dtc = tree.DecisionTreeClassifier(min_samples_leaf=10)
#dtc = tree.DecisionTreeClassifier(min_samples_split=60)
#dtc = tree.DecisionTreeClassifier(max_depth=3)

X_train, X_test, y_train, y_test = train_test_split(df_data, df_target,
                                                    test_size = 0.3,
                                                    random_state=42)

dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('R^2', dtc.score(X_test, y_test))
print('RMSE: \n', rmse)

#####ploting with pdf
import pydotplus 
tree.export_graphviz(dtc, out_file='tree.dot',feature_names=iris.feature_names)  

dot_data = tree.export_graphviz(dtc, out_file=None, feature_names=iris.feature_names) 
#dot_data = tree.export_graphviz(dtc, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
#pdf in spyder
graph.write_pdf("iris2.pdf") 
