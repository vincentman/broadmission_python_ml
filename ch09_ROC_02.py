# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:58:37 2017

@author: vincentkao
"""

#######ROC with LogisticRegression
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

iris = datasets.load_iris()
df_data = pd.DataFrame(iris['data'], columns=iris.feature_names)
df_target = pd.DataFrame(iris['target'], columns=['Target'])
df = pd.concat([df_data,df_target],axis=1)

logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(df_data[0:100],
                                                    df_target[0:100],
                                                    test_size = 0.4,
                                                    random_state=42)
logreg.fit(X_train, y_train['Target'])
y_pred_prob = logreg.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test['Target'], y_pred_prob)
print('roc_auc_score for y_pred_prob: ', roc_auc_score(y_test, y_pred_prob))
print('fpr,tpr,thresholds of roc for y_pred_prob:\n', (fpr, tpr, thresholds))

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


from sklearn.metrics import confusion_matrix, classification_report

y_pred = logreg.predict(X_test)

print('confusion_matrix for y_pred:\n', confusion_matrix(y_test, y_pred))
print('classification_report for y_pred:\n', classification_report(y_test, y_pred))
print('roc_auc_score for y_pred: ', roc_auc_score(y_test, y_pred))
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
print('fpr,tpr,thresholds of roc for y_pred:\n', (fpr, tpr, thresholds))