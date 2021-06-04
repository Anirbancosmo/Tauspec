#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: anirbanroy
"""
import numpy as np
import cosmolopy.perturbation as cp
import cosmolopy.distance as cd
from scipy.interpolate import interp1d
from scipy.integrate import quad, simps
import inputs as inp

import warnings
warnings.filterwarnings("ignore")

#Constants
c_in_m=3e8
k_bi_mec2=1.689e-10 #K^{-1}
np0=0.22*(3.086e+22)**3
sigma_t=6.652e-29/(3.086e+22)**2  #MPc^2
sigmat_times_np0_square=(sigma_t*np0)**2
pi=np.pi

#reion params
reio_model=inp.reio_model
z_re = inp.z_re
delta_z = inp.delta_z

#nuisance parameters
R_int_min=0.1 #Mpc
R_int_max=500 #Mpc\
zint_min=5
zint_max=20

#Read parameters from input file
Rb=inp.Rb
sigma_lnr=inp.sigma_lnr
b=inp.b

#Output
output_quantity= inp.output
z_pxe=inp.z_pxe
k_pxe=inp.k_pxe
ell=inp.ell

#Setup Cosmological functions and interpoler
cosmo = {'omega_M_0' : inp.omega_m_0, 'omega_lambda_0' : inp.omega_lambda, 'omega_b_0' : inp.omega_b_0, 'omega_k_0' : inp.omega_k_0, 
         'omega_n_0' : inp.omega_n_0, 'N_nu' : inp.N_nu, 'h' : inp.h, 'n' : inp.ns,'sigma_8' : inp.sigma8, 'baryonic_effects' : inp.include_barryonic_effects}


# Make chi array and redshift array for interpolation
chi=lambda z: cd.comoving_distance(z,**cosmo)
z0=np.logspace(np.log10(zint_min),np.log10(zint_max),num=100)
chi0=chi(z0)


def z_for_chi(chi_input):
    return np.interp(chi_input, chi0,z0)
 
#scale factor
def a(chi):
    return np.interp(chi, chi0,(1.0+z0)**(-1))
 
# Matter power spectra
def mpk(k,z):
    return cp.power_spectrum(k,z,**cosmo)

# ionization fraction
def xe(z):
    if(reio_model=='tanh'):
        delta_y=1.5*np.sqrt(1+z_re)*delta_z
        y=(1 + z)**(3.0/2)
        y_re=(1 + z_re)**(3.0/2)
        xe=0.5 * (1 + np.tanh((y_re - y)/delta_y))
        return xe
    
    if(reio_model=='custom'):
        reio_file=np.loadtxt(inp.reio_file_name)
        z_file=reio_file[:,0]
        xe_file=reio_file[:,1]
        xe_int=interp1d(z_file, xe_file)
        return xe_int(z)
    
    

chi_min_integ=chi(zint_min)
chi_max_integ=chi(zint_max)


def P(R):
    return (1./(R*np.sqrt(pi*sigma_lnr**2)))*np.exp(-(np.log(R/Rb)**2)/(2*sigma_lnr**2))

def W(R,k):
    return (3./(k*R)**3)*(np.sin(k*R)-(k*R)*np.cos(k*R))


def V(R):
    return 1.334*pi*R**3 


def V_av():
    integrand=lambda R: P(R)*V(R)
    res=quad(integrand, R_int_min, R_int_max)
    return res[0]


def Ik(k):
    I_integrand= lambda R, k: P(R) * V(R)* W(R,k)
    I_int =quad(I_integrand, R_int_min, R_int_max, args=(k))[0]
    return (I_int*b/V_av())
    
I=np.vectorize(Ik)      

def Fv(k):
    de_integrand=lambda R, k: P(R) * V(R)**2* W(R,k)**2
    de_int =quad(de_integrand, 0.1,500, args=(k))[0]
    return (de_int/V_av())


F=np.vectorize(Fv)

def G(k,z):
    '''Approximation used. Eq. (35) of the paper:
       https://arxiv.org/pdf/astro-ph/0511141.pdf
    '''
    mpk_at_z=mpk(k,z)
    sigma_r_square=lambda z: cp.sigma_r(Rb, z, **cosmo)[0]
    V_times_sigma=V_av() * sigma_r_square(z)
    return mpk_at_z * V_times_sigma / np.sqrt(mpk_at_z**2 + (V_times_sigma)**2)

def p_xexe_chi(chi,k):
    zin=z_for_chi(chi)
    xe_at_z=xe(zin)
    p=xe_at_z*(1-xe_at_z)*(F(k)+G(k,zin))+((1-xe_at_z)*np.log(1-xe_at_z)*I(k)-xe_at_z)**2*mpk(k,zin)
    return p

def p_xexe(zin,k):
    xe_at_z=xe(zin)
    p=xe_at_z*(1-xe_at_z)*(F(k)+G(k,zin))+((1-xe_at_z)*np.log(1-xe_at_z)*I(k)-xe_at_z)**2*mpk(k,zin)
    return p


def cltau(ell):
    chiarr=np.logspace(np.log10(int(chi_min_integ+1)),np.log10(int(chi_max_integ-1)),num=50)
    chi_len=len(chiarr)
    ellmat=np.tile(ell,(chi_len,1))
    kmat=ellmat.T/chiarr
    #pxe=p_xexe(chiarr,kmat,Rb, sigma_lnr,b)
    int_mat=((sigma_t*np0)**2/(chiarr**2*a(chiarr)**4))*p_xexe_chi(chiarr,kmat)
    res=simps(int_mat,chiarr,axis=1)
    return res
    

# save outputs in files as mentioned in the input file. 
def return_output(output):
    if 'pxe' in output:
        if z_pxe is None:
            z=z_re
        else:
            z= z_pxe
        
        if k_pxe is None:
            k=np.logspace(-2, 1)
        else:
            k=k_pxe
            
        pxe = p_xexe(z,k)
        
        f=open("pxe_z%2.2f.dat" %(z), 'w+')
        f.write("#k \t\t\t\t\t\t\t Pxe \n")
        for i in range(len(k)):
            f.write("%e \t\t\t %e \n" %(k[i], pxe[i]))
        f.close()
                
        
    if 'cltau' in output:
        if ell is None:
            l= np.logspace(np.log10(2), np.log10(5000), num=20)
        else:
            l=ell
            
        cltau_cal = cltau(l)
        
        f=open("cltau.dat", 'w+')
        f.write("# ell \t\t\t Cltau \n")
        for i in range(len(l)):
            f.write("%e \t\t\t %e \n" %(l[i], cltau_cal[i]))
        f.close()
        
    if 'xe' in output:
        z=np.linspace(0.0, 20, num=20)
        xe_cal=xe(z)
        
        f=open("xe.dat", 'w+')
        f.write("# z \t\t\t\t\t xe \n")
        for i in range(len(z)):
            f.write("%e \t\t\t %e \n" %(z[i], xe_cal[i]))
        f.close()
        
if inp.write_output:   
    return_output(output_quantity)      