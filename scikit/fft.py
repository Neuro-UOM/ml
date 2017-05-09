from numpy import genfromtxt
y= genfromtxt('../test_data/left-only-data.csv',delimiter=',')
from scipy.fftpack import fft

yf=fft(y)
# n= number of points of a column, yf is a 14 columned 2d array
n=yf[:,0].size

import numpy as np
# t = resolution of x axis
t=0.1
x=np.linspace(0.0, n*t,n)

import matplotlib.pyplot as plt
for i in range(0,yf[0].size):
    plt.plot(x,yf[:,i],label='${i}$')

#plt.legend(loc='best')
plt.grid()
plt.show()
