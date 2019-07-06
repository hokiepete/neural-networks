# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 14:04:57 2019

@author: pnola
"""
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

#This section assumes the T components are linear functions
# Set up the grid
dim = 101
a = np.linspace(0,1,dim)
da = a[1] - a[0]
a1,a2 = np.meshgrid(a,a)

#Generate T component fields
T1 = -a1 + a2
T2 = a1 + 2*a2
T3 = 3*a1 + 4*a2

#Compute the gradients of T
dT1da2,dT1da1 = np.gradient(T1,da)
dT2da2,dT2da1 = np.gradient(T2,da)
dT3da2,dT3da1 = np.gradient(T3,da)

#Allocate memory for the reconstruction estimates of A
a1_recon = np.empty([dim,dim])
a2_recon = np.empty([dim,dim])

for i in range(dim):
    for j in range(dim):
            Gradient_T = np.array([[dT1da1[i,j],dT1da2[i,j]],[dT2da1[i,j],dT2da2[i,j]],[dT3da1[i,j],dT3da2[i,j]]])
            T = np.array([T1[i,j],T2[i,j],T3[i,j]])
            #Solve the overdetermined system using least-squares approximation
            A_recon = np.linalg.lstsq(Gradient_T,T)[0]
            a1_recon[i,j]=A_recon[0]
            a2_recon[i,j]=A_recon[1]
            
#Calculate the Absolute Error
a1_error = abs(a1_recon - a1)
a2_error = abs(a2_recon - a2)

#Visualize error
plt.figure()
plt.subplot(211)
plt.pcolormesh(a1,a2,a1_error)
plt.xlabel('a1')
plt.ylabel('a2')
plt.title("a1 reconstruction error")
plt.colorbar()
plt.subplot(212)
plt.pcolormesh(a1,a2,a2_error)
plt.xlabel('a1')
plt.ylabel('a2')
plt.title("a2 reconstruction error")
plt.colorbar()    


# Set up the grid
a = np.linspace(0,1,dim)
da = a[1] - a[0]
a1,a2 = np.meshgrid(a,a)

#Generate T component fields
T1 = -a1 + a2**2
T2 = a1 + 2*a2**2
T3 = 3*a1 + 4*a2

#Compute the Derivatives of T
dT1da2,dT1da1 = np.gradient(T1,da)
dT1da2a2,dT1da1a2 = np.gradient(dT1da2,da)

dT2da2,dT2da1 = np.gradient(T2,da)
dT2da2a2,dT2da1a2 = np.gradient(dT2da2,da)

dT3da2,dT3da1 = np.gradient(T3,da)
dT3da2a2,dT3da1a2 = np.gradient(dT3da2,da)

#Allocate memory for the reconstruction estimates of A
a1_recon = np.empty([dim,dim])
a2_recon = np.empty([dim,dim])
for i in range(dim):
    for j in range(dim):
            Matrix = np.array([
                    [dT1da1[i,j],dT1da2[i,j],0.5*dT1da2a2[i,j]],
                    [dT2da1[i,j],dT2da2[i,j],0.5*dT2da2a2[i,j]],
                    [dT3da1[i,j],dT3da2[i,j],0.5*dT3da2a2[i,j]]])
            T = np.array([T1[i,j],T2[i,j],T3[i,j]])
            #Solve, system is fully determined
            A_recon = np.linalg.solve(Matrix,T)
            a1_recon[i,j]=A_recon[0]
            a2_recon[i,j]=A_recon[1]

#Calculate the Absolute Error            
a1_error = abs(a1_recon - a1)
a2_error = abs(a2_recon - a2)

#Visualize
plt.figure()
plt.subplot(211)
plt.pcolormesh(a1,a2,a1_error)
plt.xlabel('a1')
plt.ylabel('a2')
plt.title("a1 reconstruction error")
plt.colorbar()
plt.subplot(212)
plt.pcolormesh(a1,a2,a2_error)
plt.xlabel('a1')
plt.ylabel('a2')
plt.title("a2 reconstruction error")
plt.colorbar()    
#'''

