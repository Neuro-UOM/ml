# -*- coding: utf-8 -*-
"""
Created on Sat May 20 11:27:23 2017

@author: dinuka
"""

from sklearn.externals import joblib
clf = joblib.load('decision_tree_model.pkl') 

import pandas as pd

location='/home/dinuka/fyp/ml_git_repo/Data/train.csv'
df=pd.read_csv(location,header=None)

arr=df.as_matrix()

from scipy.fftpack import fft
y=fft(df)

#print clf.predict(y[3700:3836,:14])


#output=clf.predict(y[-200:,:14])

from sklearn import metrics
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(clf, y[:,:14], arr[:,-1], cv=10)
print metrics.accuracy_score(arr[:,-1], predicted) 