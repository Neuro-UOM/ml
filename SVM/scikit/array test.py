# -*- coding: utf-8 -*-
"""
Created on Fri May 19 19:40:02 2017

@author: dinuka
"""

location='/home/dinuka/fyp/ml_git_repo/Data/train.csv'
df=pd.read_csv(location,header=None)
#columns=list(df)

yf=df
y=df.as_matrix()

x= y[:,-1]
labels=[]

for i in range(x.size):
    if(x[i]==1):
        labels.insert(i,'left')
    else:
        labels.insert(i,'right')

print labels