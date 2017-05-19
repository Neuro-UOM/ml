#from numpy import genfromtxt
#y= genfromtxt('/home/dinuka/fyp/our_data/left-only-data.csv',delimiter=',')

import pandas as pd

location='/home/dinuka/fyp/ml_git_repo/Data/train.csv'
df=pd.read_csv(location,header=None)
#columns=list(df)

yf=df[:]

test_data=df[3700:3836].append(df[-200:])

for i in range(3700,3836):
    yf.drop(i,inplace=True)

for i in range((len(yf)-200),len(yf)):
    yf.drop(i,inplace=True)

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


# n= number of points of a column, yf is a 14 columned 2d array
#n=yf[:,0][1:100].size
#print yf[:,0][1:100]

#import matplotlib.pyplot as plt
#
#plt.plot(x,yf[:,0][1:100])
#plt.grid()
#plt.show()

import numpy as np
# t = resolution of x axis
#t=0.1
#x=np.linspace(0.0, n*t,n)

from sklearn import svm
clf = svm.SVC() #kernel='linear'
yf2=yf[:,0:14]
#clf.fit(x[:5000],yf[:5000,-1])
clf.fit(yf2,labels)


# to save the built model
from sklearn.externals import joblib
joblib.dump(clf, 'svm_model.pkl')
