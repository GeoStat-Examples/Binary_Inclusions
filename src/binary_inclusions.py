#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 17:09:00 2020

@author: zech0001
"""

import numpy as np
import copy


parameters_simple_DEF = dict(
        k_bulk      = 1e-5, #conductivity value of bulk
        k_incl      = 1e-3, #conductivity value of inclusions
        lx          = 10,   #inclusion length in x-direction
        ly          = 10,   #inclusion length in y-direction
        lz          = 0.5,  #inclusion length in z-direction
        nx          = 5,    #number of inclusions in x-direction
        ny          = 3,    #number of inclusions in y-direction
        nz          = 10,   #number of inclusions in z-direction
        nz_incl     = 3,    #number of inclusions within different Kvalue in z-block 
#        ratio       = 0.2,  #ratio of conductivities/amount of inclusions (K_incl)
        )

parameters_block_DEF = dict(
        k_bulk      = [1e-5,1e-3],  #conductivity value of bulk
        k_incl      = [1e-3,1e-5],  #conductivity value of inclusions
        nn          = [5,5],        #number of inclusions per block 
        ll          = [10,10],      #inclusion length 
        nn_incl     = [3,3],        #number of inclusions with different Kvalue 
        )


class Simple_Binary_Inclusions():
    
    def __init__(self,dim=2,**parameters):

        self.parameters=copy.copy(parameters_simple_DEF) 
        self.parameters.update(parameters)   
        
        if dim!=2 and dim!=3:
            raise ValueError('Routine implemented for 2D and 3D.')
        self.dim=dim
    
    def structure(self,seed=None,bimodalKeff=False):
      
        """ 
        Create binary structure in 2D or 3D with bulk conductivity 'k_bulk' and a number 
        of inclusions of conductivity which form small blocks within the bulk 
               
        Return
        ------
        
        """
        if seed != None:        
            np.random.seed(seed)

        if bimodalKeff:
             self.bimodal_Keff(self.parameters['k_bulk'],self.parameters['k_incl'])

        if self.dim ==2:    
            kk=self.parameters['k_bulk']*np.ones([self.parameters['nx'],self.parameters['nz']])

            for ix in range(self.parameters['nx']):    
         
                zz_index=random_unique(self.parameters['nz'],size=self.parameters['nz_incl'])
                for iz,zz in enumerate(zz_index):
                    kk[ix,zz]=self.parameters['k_incl']
    
        elif self.dim ==3:    
            kk=self.parameters['k_bulk']*np.ones([self.parameters['nx'],self.parameters['ny'],self.parameters['nz']])
            
            for iy in range(self.parameters['ny']):    
            ### index of y-compartment ii = in blocks 0 - Ny       
                for ix in range(self.parameters['nx']):    
                ### index of x-compartment jj = in blocks 0 - Nx       

                    zz_index=random_unique(self.parameters['nz'],size=self.parameters['nz_incl'])
                    for iz,zz in enumerate(zz_index):
                        kk[ix,iy,zz]=self.parameters['k_incl']
            
        self.kk=kk            

        return self.kk

    def structure2scale(self,x0=0,y0=0,z0=0,endpoint=True):

        if endpoint:        
            self.x = x0+np.linspace(0,self.parameters['nx']*self.parameters['lx'],self.parameters['nx']+1,endpoint=True)
            self.y = y0+np.linspace(0,self.parameters['ny']*self.parameters['ly'],self.parameters['ny']+1,endpoint=True)
            self.z = z0+np.linspace(0,self.parameters['nz']*self.parameters['lz'],self.parameters['nz']+1,endpoint=True)
        else:
            self.x = x0+np.arange(0,self.parameters['nx']*self.parameters['lx'],self.parameters['lx'])
            self.y = y0+np.arange(0,self.parameters['ny']*self.parameters['ly'],self.parameters['ly'])
            self.z = z0+np.arange(0,self.parameters['nz']*self.parameters['lz'],self.parameters['lz'])

    def scales2mesh(self,mesh,**kwargs):

        self.structure2scale(**kwargs,endpoint=False)

        index=np.zeros_like(mesh,dtype=int)                                   
        for ii in range(len(mesh)):
            index[ii,0]=find_nearest_index(mesh[ii,0], self.x+0.5*self.parameters['lx'])
            index[ii,1]=find_nearest_index(mesh[ii,1], self.y+0.5*self.parameters['ly'])
            index[ii,2]=find_nearest_index(mesh[ii,2], self.z+0.5*self.parameters['lz'])
        self.index = index
        
        return self.index
              
    def structure2mesh(self):
    
        '''
        Creats inclusion structure on a specified mesh given as array of x,y, z - coordinates 
        '''            
            
        if self.dim ==2:    
            self.kk_mesh=self.kk[self.index[:,0],self.index[:,2]]
        elif self.dim ==3:    
            self.kk_mesh=self.kk[self.index[:,0],self.index[:,1],self.index[:,2]]

        return self.kk_mesh

    def bimodal_Keff(self,k1,k2):

        """
        Calculates conductivity values of bulk and inclusions 
        from average effective values based on binary statistics    
        """
        pp = self.parameters['nz_incl']/self.parameters['nz']
        #print (pp,(1.-pp)/(1.-2.*pp),pp/(1.-2.*pp))   

        k_bulk = k1**((1.-pp)/(1.-2.*pp))/k2**(pp/(1.-2.*pp))    
        k_incl = k2**((1.-pp)/(1.-2.*pp))/k1**(pp/(1.-2.*pp))    
    
        self.parameters.update(
                k_bulk  = k_bulk,
                k_incl  = k_incl,
                ratio   = pp,
                var     = pp*(1.-pp)*(np.log(k1)-np.log(k2))**2,
                )
    
        return k_bulk,k_incl

class Block_Binary_Inclusions():
    
    def __init__(self,
                 dim=2,
                 axis = 0, 
                 **parameters):

        self.parameters=copy.copy(parameters_simple_DEF) 
        self.parameters.update(parameters_block_DEF)         
        self.parameters.update(parameters)   
        self.nn = self.parameters['nn']
        self.axis = axis

        if dim!=2 and dim!=3:
            raise ValueError('Routine implemented for 2D and 3D.')
        self.dim=dim
    
    def structure(self,seed=None,bimodalKeff=False):
      
        """ 
        Create binary structure in 2D or 3D with bulk conductivity 'k_bulk' and a number 
        of inclusions of conductivity which form small blocks within the bulk 
               
        Return
        ------
        
        """
               
        if seed != None:        
            np.random.seed(seed)

        blocks = []
        for ii in range(len(self.nn)):

            BIS=Simple_Binary_Inclusions(**self.parameters)
            if self.axis == 0:
                BIS.parameters.update(
                        nx = self.parameters['nn'][ii],
                        lx = self.parameters['ll'][ii],
                        )
            elif self.axis == 1:
                BIS.parameters.update(
                        ny = self.parameters['nn'][ii],
                        ly = self.parameters['ll'][ii],
                        )
            elif self.axis == 2:
                BIS.parameters.update(
                        nz = self.parameters['nn'][ii],
                        lz = self.parameters['ll'][ii],
                        )
            BIS.parameters.update(
                    nz_incl = self.parameters['nn_incl'][ii],
                    k_bulk  = self.parameters['k_bulk'][ii],
                    k_incl  = self.parameters['k_incl'][ii],
                    )

            blocks.append(BIS.structure(bimodalKeff=bimodalKeff))

        kk = blocks[0]
        for i in range(len(blocks)-1):
            kk   = np.concatenate((kk,blocks[i+1]),axis=self.axis)

        self.kk = kk
        return self.kk

    def structure2scale(self,x0=0,y0=0,z0=0,endpoint=True):

        if endpoint:                   
            self.x = x0+np.linspace(0,self.parameters['nx']*self.parameters['lx'],self.parameters['nx']+1,endpoint=True)
            self.y = y0+np.linspace(0,self.parameters['ny']*self.parameters['ly'],self.parameters['ny']+1,endpoint=True)
            self.z = z0+np.linspace(0,self.parameters['nz']*self.parameters['lz'],self.parameters['nz']+1,endpoint=True)
        else:
            self.x = x0+np.arange(0,self.parameters['nx']*self.parameters['lx'],self.parameters['lx'])
            self.y = y0+np.arange(0,self.parameters['ny']*self.parameters['ly'],self.parameters['ly'])
            self.z = z0+np.arange(0,self.parameters['nz']*self.parameters['lz'],self.parameters['lz'])

        c0 = [0]
        for i in range(len(self.nn)):
            c1 = c0[-1]+np.linspace(0,self.nn[i]*self.parameters['ll'][i],self.nn[i]+1,endpoint=True)
            c0 = np.concatenate((c0,c1[1:]))

        if self.axis == 0:
            self.x = x0+c0
        elif self.axis == 1:
            self.y = z0+c0
        elif self.axis == 2:
            self.z = y0+c0

    def scales2mesh(self,mesh,**kwargs):

        self.structure2scale(**kwargs,endpoint=True)

        cx = self.x[:-1] + 0.5*np.diff(self.x)
        cy = self.y[:-1] + 0.5*np.diff(self.y)
        cz = self.z[:-1] + 0.5*np.diff(self.z)

        index=np.zeros_like(mesh,dtype=int)                                   
        for ii in range(len(mesh)):
            index[ii,0]=find_nearest_index(mesh[ii,0], cx)
            index[ii,1]=find_nearest_index(mesh[ii,1], cy)
            index[ii,2]=find_nearest_index(mesh[ii,2], cz)
        self.index = index
        
        return self.index
              
    def structure2mesh(self):
    
        '''
        Creats inclusion structure on a specified mesh given as array of x,y, z - coordinates 
        '''            
            
        if self.dim ==2:    
            self.kk_mesh=self.kk[self.index[:,0],self.index[:,2]]
        elif self.dim ==3:    
            self.kk_mesh=self.kk[self.index[:,0],self.index[:,1],self.index[:,2]]

        return self.kk_mesh


def random_unique(n,size=1):

    """  Return array of length 'size', containing unique integer random 
         numbers between 0 and N

         Return random integers from the "discrete uniform" distribution of
         in the half-open interval [0, N). 

         Parameters    
         ----------
         N          : int    
                      one above the largest (signed) integer to be drawn from
         size       : int, optional
                      length of array
        
        Returns
        ------
        aa_unique   : array
                      array of unique random values 
    
    """
    
    aa = np.random.randint(n,size=2*size)
    aa_unique=unique_unsorted(aa)
    
    i=1
    while len(aa_unique)<size:
        aa = np.random.randint(n,size=2*i*size)
        aa_unique=unique_unsorted(aa)
        i+=1
        
    return aa_unique[:size]

def unique_unsorted(aa):

    """
        Return unique values of aa in order of appearance
        Values are not sorted as done by unique
        
        Parameters    
        ----------
        aa          : array
                      array of values
 
        Returns
        ------
        aa_unique   : array 
                      aa reduced to the unique values (unsorted)
    """

    uu,indeces = np.unique(aa, return_index=True)
    aa_unique= [aa[index] for index in sorted(indeces)]

    return aa_unique

def find_nearest_index(value, array):

    """Find index of nearest match of given 'value' in given 'array'.

        Parameter
        ---------
        
        value       :   float    
                        index of this number will be found in array
        array       :   array
                        search for nearest match in this numpy array
        
        Return
        ------
        index       :   int
                        index of entry in 'array' being closest to 'value'
    """

    index = (np.abs(array-value)).argmin()
    return index

