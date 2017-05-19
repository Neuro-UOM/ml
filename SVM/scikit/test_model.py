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

print clf.predict(y[3500:4000,:14])