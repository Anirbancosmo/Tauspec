#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: anirbanroy
"""
import numpy as np

###############################################################################
# Cosmology
###############################################################################
'''
Check CosmoloPy documentation for understanding the names of cosmological parameters
LINK:  https://roban.github.io/CosmoloPy/
'''

h=0.69
ns=0.9667
s8=0.8
omega_b_0= 0.04757
omega_lambda= 0.721
omega_m_0= 0.2877
omega_k_0= 0.0
omega_n_0= 0.0
N_nu = 0.0
include_barryonic_effects= False
sigma8=0.80


###############################################################################
# Reio History
###############################################################################

'''
If reio model is tanh, mention redshift (z_re) and duration (delta_z) of 
reionization. If reio model is custom , provide file location of text/ dat file having
z (first column) and xe (second column)
'''
reio_model= "tanh" # or custom
z_re=7.5  # redshift of reionization ( when xe=0.5)
delta_z=2 #duration of reionization

reio_file_name=None 
#reio_file_name='/Users/anirbanroy/Documents/Tauspec/xe.dat'

###############################################################################
# Patchy Reionization Model
###############################################################################

Rb = 5 #characteristic bubble size (Mpc)
sigma_lnr= np.log(2) #bubble size distribution
b = 6 # linar bias 
T0 = 2e4  # in Kelvin

###############################################################################
# Output
###############################################################################
'''
If write_output=True, outputs will be saved in files. 
pxe: power spectra of ionization fraction
cltau: optical depth power spectrum
xe: ionization power spectrum
'''

write_output=True
output= {'pxe', 'cltau', 'xe'}
#if output pxe, mention z_pxe (scalar) and k_pxe (array) range
z_pxe= None
k_pxe= None
#if output cltau, mention multiplole range (array), defaulte ell is (2, 5000)
ell= None