# -*- coding: utf-8 -*-
"""
Created on Sat May 20 11:48:47 2017

@author: dinuka
"""

import pandas as pd

location='/home/dinuka/fyp/ml_git_repo/Data/train.csv'
df=pd.read_csv(location,header=None)
#columns=list(df)

yf=df[:]

y=yf.as_matrix()
x= y[:,-1]

labels=[]

for i in range(x.size):
    if(x[i]==1):
        labels.insert(i,'left')
    else:
        labels.insert(i,'right')

from scipy.fftpack import fft
yf=fft(yf)



from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

yf2=yf[:,:-1]
#clf.fit(x[:5000],yf[:5000,-1])
#clf.fit(yf2,x)

y_pred = gnb.fit(yf2, x).predict(yf2)
clf=gnb
#clf.fit(y[:,:-1],labels)

# to save the built model
from sklearn.externals import joblib
joblib.dump(clf, 'nb_gaussian_model.pkl')
