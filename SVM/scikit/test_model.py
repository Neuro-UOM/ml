# -*- coding: utf-8 -*-
"""
Created on Fri May 19 17:05:08 2017

@author: dinuka
"""

from sklearn.externals import joblib
clf = joblib.load('svm_model.pkl') 

import pandas as pd

location='/home/dinuka/fyp/ml_git_repo/Data/train.csv'
df=pd.read_csv(location,header=None)

from scipy.fftpack import fft
y=fft(df)

#print clf.predict(y[3700:3836,:14])


output=clf.predict(y[-200:,:14])

c=0
for i in output:
    if(i=='left'):
        c+=1
        
print 'accuracy', (200-c)/200.0*100,'%'
