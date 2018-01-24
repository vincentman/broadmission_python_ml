# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:02:50 2017

@author: vincentkao
"""

#PCA
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, :2]  
Y = iris.target

X_reduced = PCA(n_components=3).fit_transform(iris.data)
plt.scatter(X_reduced[Y==0, 0], np.zeros((50,1))+0.02, color='red', marker='^')
plt.scatter(X_reduced[Y==1, 0], np.zeros((50,1))-0.02, color='blue', marker='x')
plt.scatter(X_reduced[Y==2, 0], np.zeros((50,1)), color='green', marker='o')
plt.set_title("First three PCA directions")
plt.set_xlabel("1st eigenvector")
plt.w_xaxis.set_ticklabels([])
plt.set_ylabel("2nd eigenvector")
plt.w_yaxis.set_ticklabels([])
plt.set_zlabel("3rd eigenvector")
plt.w_zaxis.set_ticklabels([])

plt.show()