#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 09:24:59 2021

@author: anirbanroy
"""


import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

# Change the directory 
Home='/Users/anirbanroy/Documents/work/Research/Project_Girish/new/ps160_sher/'

# the redshifts at which power spectra is calculated
zall = [15.19, 13.75, 12.59, 11.63, 10.83, 10.14, 9.55, 9.02, 8.56, 8.15, 7.78, 7.44, 7.14, 6.86, 6.60, 6.37, 6.15, 5.95, 5.76, 5.58, 5.41, 5.26, 5.11]


def read_file():
    '''
    The function which reads the file of xe and it is later used for 
    interpolation to get power spectra at any redshift.
    
    Returns: k and p(k)
    '''
    k=[]
    pk=[]
    
    # Read one file first and know the structure of the file. 
    f=np.loadtxt(Home + "ps_L160N2048_sherwood_z%.2f.txt" %(zall[0]))
    lenf=len(f[:,0][1:])
    lenz=len(zall)
    kf=np.zeros((lenz,lenf))   
    pkf=np.zeros((lenz,lenf))
        
    for i in range(lenz):
        f=np.loadtxt(Home + "ps_L160N2048_sherwood_z%.2f.txt" %(zall[i]))
        
        k=f[:,0][1:]
        pk=f[:,1][1:]        
        kf[i]=k
        pkf[i]=pk
    return kf[0], pkf


# Get all quantities for the interpolation
kint, pkint= read_file()
f=interp2d(kint,zall,pkint,kind='cubic') 


def get_pk_at_z(z_req):
    '''
    This function returns k and p(k) at any redshift. 
    Reshift should be in the range of 15.19 to 5.11.
    '''
    
    pk_res=f(kint,z_req)
    return kint, pk_res


def test_plot(zin, quantity="dk"):
    kp, pkp=get_pk_at_z(zin)
    if quantity=="dk":
        res= kp**3*pkp/2.0/np.pi**2
        ylabel=r'$\Delta_{\rm x_e x_e}(k)$'
    else:
        res= pkp
        ylabel=r'$P_{\rm x_e x_e} (k)$'
    
    plt.loglog(kp, res)
    plt.title("z=%.2f" %zin)
    plt.xlabel(r"$k$")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()
    
    
    
    