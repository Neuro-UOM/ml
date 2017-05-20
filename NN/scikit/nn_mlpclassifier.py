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



from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

yf2=yf[:,:-1]
#clf.fit(x[:5000],yf[:5000,-1])
#clf.fit(yf2,x)

clf.fit(y[:,:-1],labels)

# to save the built model
from sklearn.externals import joblib
joblib.dump(clf, 'nn_mlp_model.pkl')
