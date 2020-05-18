#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: suriyaprakashjambunathan
"""
import numpy as np
from numpy import linalg as LA
import math as math
import matplotlib.pyplot as plt
x=np.array([[0.59,1.67,1.42,1.11,1.64,0.39,1.42,1.43,1.45,1.45,0.70,1.21],[0.59,1.73,1.10,1.09,1.79,0.66,1.52,1.52,1.54,1.54,0.81,1.31]])
mean=np.zeros((2,1))
cov_comp=np.zeros((2,2))

#Co-variance calculation
for i in range(0,12):
    y_comp=np.array([[x[0][i]],[x[1][i]]])
    mean=mean+y_comp
    cov_comp=cov_comp+y_comp.dot(y_comp.transpose())
mean=mean/12.0
cov_comp=cov_comp/12.0
cov=cov_comp-mean.dot(mean.transpose())
print('C=',cov)
w, v = LA.eig(cov)
print('D=',w)
print('E=',v)
v=v.transpose()
y=(v).dot(x)
print('y=',y)
#Reconstruction
x1=v.transpose().dot(y)
x2=np.zeros((2,12))
x2[0]=y[0]*v[0][0]
x2[1]=y[0]*v[0][1]
print('x1=',x1)
print('x2=',x2)

# Distance 
m=v[0][1]/v[0][0]
r=math.sqrt(1+m**2)
d=np.zeros((12))
for i in range(0,12):
    d[i]=math.sqrt((x[1][i]**2 + x[0][i]**2)-(abs(x[1][i]-m*x[0][i])/r)**2)
print('y1=',d)
print('D1=',d)

#GRAPHS
z=np.zeros((1,12))
n= np.linspace(0.0,2.5,1000)
plt.scatter(y[0],y[1],color='k')
plt.scatter(x[0],x[1],color='g')
m=v[0][1]/v[0][0]
plt.plot(n,m*n,color='r')
plt.plot(n,(-1)*n/m,color='r')
plt.grid()
plt.show()
