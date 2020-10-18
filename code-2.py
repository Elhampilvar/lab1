#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu


mu, sigma = 0, 1 # mean and standard deviation

def ran_gen(n):
    s = np.random.normal(mu, sigma, n*n)
    return s.reshape(n,n)

def GEPP(A):
    '''
    Gaussian elimination with partial pivoting.

    '''
    n = len(A)
    doPricing = True
    for k in range(n-1):
        for row in range(k+1, n):
            multiplier = A[row,k]/A[k,k]
            A[row, k:] = A[row, k:] - multiplier*A[k, k:]
    return A

n= []
gf = []

for i in range(8,1024,2):
    for q in range(2):
        z = ran_gen(i)
        d = max(z.max(), np.abs(z.min()))
        L, U, z = lu(z)
        #z = GEPP(z)
        u = max(z.max(), np.abs(z.min()))
        gf.append(u/d)
        n.append(i)
    
#fitting part
n_log = np.log(n)
gf_log = np.log(gf)
coef = np.polyfit(n_log, gf_log, 1)
plt.figure(figsize =[10,10])
plt.plot(n, gf, '*r', label = 'data')
plt.plot(n, np.exp(coef[1])*n**coef[0], '-b', label = '$\\alpha$='+str(coef[0]))
plt.plot(range(1,8),[2**float(i) for i in range(1,8)],'--y', label = '$2^{n}$')
plt.legend()
plt.xscale('log', basex=2)
plt.yscale('log', basey=2) 
plt.ylabel("Growth Factor")
plt.xlabel("Size of Matrix")

#histgram for fixed n
plt.figure(figsize =[10,10])
n=16
gf=[]
for i in range(4000):
    z = ran_gen(n)
    d = max(z.max(), np.abs(z.min()))
    L, U, z = lu(z)
    #z = GEPP(z)
    u = max(z.max(), np.abs(z.min()))
    gf.append(u/d)
    
n, bins, patches = plt.hist(gf, 100, density=True, facecolor='b', edgecolor='r')    
plt.xlabel('Growth Factor')
plt.ylabel('Probability Density Function')
plt.title('Histogram of Growth Factor for n= 16')
plt.grid(True)
plt.show()

