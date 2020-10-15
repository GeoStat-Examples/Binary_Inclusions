#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 13:15:07 2020

@author: zech0001
"""

import numpy as np
import matplotlib.pyplot as plt
import binary_inclusions as bi

np.random.seed(20201012+9) #20201012+2

BI=bi.Simple_Binary_Inclusions(
        dim     = 2,
        k_bulk  = 1e-5, # conductivity value of bulk
        k_incl  = 1e-3, # conductivity value of inclusions
        nx      = 4,    # number of inclusions in x-direction
        lx      = 10,   # inclusion length in x-direction
        nz      = 20,   # number of inclusions in z-direction
        lz      = 0.5,  # inclusion length in z-direction
        nz_incl = 3,    # number of inclusions with different K 
        )

###########################################
### Simple 2D binary inclusion structure

test =BI.bimodal_Keff(k1 = 2e-4,k2=2e-6)
kk=BI.structure()
BI.structure2scale()
xx,zz = np.meshgrid(BI.x,BI.z)

fig = plt.figure(1)  
ax = fig.add_subplot(1,1,1)

im = ax.pcolor(xx,zz,kk.T,cmap=plt.get_cmap('binary_r'))
ax.set_xlabel('$x$ [m]') 
ax.set_ylabel('$z$ [m]')   

###########################################
### Two stage 2D binary inclusion structure

BIB=bi.Block_Binary_Inclusions(
        dim         = 2,
        axis        = 0,
        k_bulk      = [1e-5,1e-3],  # conductivity value of bulk
        k_incl      = [1e-3,1e-5],  # conductivity value of inclusions
        nn          = [4,20],       # number of inclusions-blocks in x-direction
        ll          = [10,10],      # inclusion length in x-direction
        nz          = 20,           # number of inclusions in z-direction
        lz          = 0.5,          # inclusion length in z-direction
        nn_incl     = [3,3],        # number of inclusions within different K 
        )

kk = BIB.structure()
BIB.structure2scale(x0=-20,z0=52)

xx,zz = np.meshgrid(BIB.x,BIB.z)

fig = plt.figure(2,figsize=[6,2])  
ax = fig.add_subplot(1,1,1)
im = ax.pcolor(xx,zz,kk.T,cmap=plt.get_cmap('binary_r'))
#im = ax.pcolor(kk.T,cmap=plt.get_cmap('binary'))
ax.set_xlabel('$x$ [m]') 
ax.set_ylabel('$z$ [m]')   
  