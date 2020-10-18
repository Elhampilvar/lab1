#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

#STEEPEST DESCENT METHOD
k=1000 #length of matrix
a = np.zeros((k,k)) #defining A matrix
for i in range(k-1):
    a[i][i] = i+1
    a[i][i+1] = 1
    a[i+1][i] = 1

a[k-1][k-1]= k
b = k*[1]      #defining b matrix
x = k*[0]      #initial solution
kk = LA.norm(a)*LA.norm(LA.inv(a)) #condition number of A matrix 
z= range(k)
norm_sd=[] # list for saving norm
phi=[]     # list for saving condition number
for i in z:
    r=b- np.matmul(a,x)
    norm_sd.append(LA.norm(r))
    alpha = np.dot(r,r)/np.dot(r,np.matmul(a,r))
    x = x + np.dot(alpha,r)
    phi.append(2*((np.sqrt(kk)-1)/(np.sqrt(kk)+1))**i)

#CONJUGATE GRADIENT METHOD
k=1000 #length of matrix
a = np.zeros((k,k)) #defining A matrix
for i in range(k-1):
    a[i][i] = i+1
    a[i][i+1] = 1
    a[i+1][i] = 1

a[k-1][k-1]= k
b = k*[1]      #defining b matrix
x = k*[0]      #initial solution

z= range(k)
norm_cg=[] # list for saving norm
norm_r=[]     # list for saving condition number

#first step i=1
beta = 0
rold = b
d = b
alpha = np.dot(rold,rold)/np.dot(d,np.matmul(a,d))
x = x + np.dot(alpha,d)
r = rold - np.dot(alpha,np.matmul(a,d))
norm_cg.append(LA.norm(r))
norm_r.append(LA.norm(b-np.matmul(a,x)))


for i in range(1,k):
    beta = np.dot(r,r)/np.dot(rold,rold)
    d = r + np.dot(beta,d)
    alpha = np.dot(r,r)/np.dot(d,np.matmul(a,d))
    x = x + np.dot(alpha,d)
    rold = r
    r = r - np.dot(alpha,np.matmul(a,d)) 
    norm_cg.append(LA.norm(r))
    norm_r.append(LA.norm(b-np.matmul(a,x)))

#plot  
plt.figure(figsize =[10,10])
plt.plot(z, norm_cg, label= "$||r_i||$ conj-gra")
plt.plot(z, norm_r, label = '$|| b-Ax_i||_2$')
plt.plot(z, phi, label= "$2(\\frac{\sqrt{k}-1}{\sqrt{k}+1})^i$")
plt.plot(z, norm_sd, label = '$|| r_i||_2$ ste-desc')
plt.yscale("log")
plt.ylim(10**(-17),10**3)
plt.xlabel("i")
plt.grid()
plt.legend(fontsize='xx-large')
plt.show()

