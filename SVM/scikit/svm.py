from numpy import genfromtxt
y= genfromtxt('/home/dinuka/fyp/our_data/left-only-data.csv',delimiter=',')
from scipy.fftpack import fft

yf=fft(y)
# n= number of points of a column, yf is a 14 columned 2d array
n=yf[:,0][1:100].size
print yf[:,0][1:100]

import numpy as np
# t = resolution of x axis
t=0.1
x=np.linspace(0.0, n*t,n)

from sklearn import svm
svc = svm.SVC(kernel='linear')
