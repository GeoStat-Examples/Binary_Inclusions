# -*- coding: utf-8 -*-
"""
Script to create and visualize random binary inclusion structures:
    - simple inclusion structure
    - block inclusion structure
Fields are by default in 2D for visualization.

Structures in 3D and different block arrangement can be created by modifying
setting parameters, particular dim and/or axis

@author: A. Zech
Licence MIT, A.Zech, 2020
"""

import numpy as np
import matplotlib.pyplot as plt
import binary_inclusions as bi

np.random.seed(20201101)

#########################################
### Simple binary inclusion structure ###
#########################################

### initialize simple binary inclusion structure with specified settings
### as instance of the class Simple_Binary_Inclusions
BI = bi.Simple_Binary_Inclusions(
    dim=2,          # dimesionality of structure
    k_bulk=1e-5,    # bulk conductivity value
    k_incl=1e-3,    # conductivity value of inclusions
    nx=8,           # number of units in x-direction
    lx=10,          # unit length in x-direction
    nz=20,          # number of unit in z-direction
    lz=0.5,         # unit length in z-direction
    nz_incl=3,      # number of inclusions (units with different K)
)

### Generate random realization of simple binary inclusion structure ###
k1 = BI.structure()
BI.structure2scale()
xx, zz = np.meshgrid(BI.x, BI.z)

### Plot random realization of simple binary inclusion structure ###
fig = plt.figure(1)
im = plt.pcolor(xx, zz, k1.T, cmap=plt.get_cmap("binary_r"))
plt.xlabel("$x$ [m]")
plt.ylabel("$z$ [m]")
plt.tight_layout()
plt.savefig("../results/BI_Simple.png", dpi=300)
print('Save figure of simple inclusion structure to ./results')

############################################
### Two block binary inclusion structure ###
############################################

### initialize block binary inclusion structure with specified settings
### as instance of the class Block_Binary_Inclusions
BIB = bi.Block_Binary_Inclusions(
    dim=2,                  # dimesionality of structure
    axis=0,                 # direction of multiple blocks (0=x, 1=z, 2=y)
    k_bulk=[1e-5, 1e-3],    # bulk-conductivity value in each block
    k_incl=[1e-3, 1e-5],    # conductivity values of inclusions in each block
    nn=[4, 18],             # number of units within each block (now x-dir)
    ll=[10, 10],            # unit lengths within blocks (now x-dir)
    nz=20,                  # number of units in z-direction
    lz=0.5,                 # unit length in z-direction
    nn_incl=[3, 3],         # number of inclusions (units within different K)
)

### Generate random realization of simple binary inclusion structure ###
k2 = BIB.structure()
BIB.structure2scale(x0=-20, z0=52)
xx, zz = np.meshgrid(BIB.x, BIB.z)

### Plot random realization of simple binary inclusion structure ###
fig = plt.figure(2, figsize=[10, 2.5])
plt.pcolor(xx,zz,k2.T,cmap=plt.get_cmap('binary_r'))
plt.xlabel("$x$ [m]")
plt.ylabel("$z$ [m]")
plt.tight_layout()
plt.savefig("../results/BI_Block.png", dpi=300)
print('Save figure of block inclusion structure to ./results')
