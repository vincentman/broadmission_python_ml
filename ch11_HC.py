#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 21:58:13 2017

@author: jerry
"""
#HC
from sklearn import cluster, datasets, metrics
import pandas as pd

iris = datasets.load_iris()

df_data = pd.DataFrame(iris['data'], columns=iris.feature_names)
df_target = pd.DataFrame(iris['target'], columns=['Target'])
hclust = cluster.AgglomerativeClustering(linkage = 'ward',
                                         affinity = 'euclidean',
                                         n_clusters = 3)

hclust.fit(df_data)
cluster_labels = hclust.labels_
print('labels:\n', cluster_labels)

silhouette_avg = metrics.silhouette_score(df_data, cluster_labels)
print('silhouette_score: ', silhouette_avg)

######
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

#mergings = linkage(df_data, method='ward', metric='euclidean')
mergings = linkage(df_data, method='complete')


dendrogram(mergings,
           labels=cluster_labels)

plt.show()


