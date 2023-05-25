#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 2023

@author: cesar
"""

###############################################################################
#   Libraries
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from argparse import Namespace
from scipy import interpolate

jpac_blue   = "#1F77B4"; jpac_red    = "#D61D28"; jpac_green  = "#2CA02C"; 
jpac_orange = "#FF7F0E"; jpac_purple = "#9467BD"; jpac_brown  = "#8C564B";
jpac_pink   = "#E377C2"; jpac_gold   = "#BCBD22"; jpac_aqua   = "#17BECF"; 
jpac_grey   = "#7F7F7F";

jpac_color = [jpac_blue, jpac_red, jpac_green, jpac_orange, jpac_purple,
              jpac_brown, jpac_pink, jpac_gold, jpac_aqua, jpac_grey, 'black' ];

dashes, jpac_axes = 10*'-', jpac_color[10];

hbarc2 = 1./(2.56819e-6);
mproton, mpsi, mphoton = 0.938272, 3.0969160, 0.;
md, mdbar, mlambdac = 1.86484, 2.00685, 2.28646; 

def kallen(x,y,z): return x*x + y*y + z*z - 2.*(x*y + x*z + y*z)

def sfromEbeam(ebeam,mp): return mp*( mp + 2.*ebeam);

def Ebeamfroms(s,mp): return (s-mp*mp)/2./mp

def momentum(s,m1,m2):
    out = kallen(s,m1**2,m2**2)
    if out > 0.:
        return np.sqrt(out)/np.sqrt(s)/2.;
    return 0;

def cmomentum(si,m1,m2):
    s = si + 1j*0.0000001
    q2 = kallen(s,m1**2,m2**2)/s/4.;
    return np.sqrt(q2)

def costhetafromt(s,t,m1,m2,m3,m4):
    p1, p3 = momentum(s,m1,m2), momentum(s,m3,m4);
    e1, e3 = np.sqrt(m1*m1+p1*p1), np.sqrt(m3*m3+p3*p3);
    return ( t - m3*m3 - m1*m1 + 2.*e1*e3 )/(2.*p1*p3);

def tfromcostheta(s,x,m1,m2,m3,m4):
    p1, p3 = momentum(s,m1,m2), momentum(s,m3,m4);
    e1, e3 = np.sqrt(m1*m1+p1*p1), np.sqrt(m3*m3+p3*p3);
    return m3*m3 + m1*m1 - 2.*e1*e3 + 2.*p1*p3*x;

def interpolation(x,y): 
    return interpolate.interp1d(x, y,kind='cubic')

###############################################################################
###############################################################################
#
#   Main program
#
###############################################################################
###############################################################################

###############################################################################
#   Uploading experimental data
###############################################################################

file_sigmagluex = "sigma_gluex.txt"; sigmagluex = np.loadtxt(file_sigmagluex);
Ebeam_sigmagluex = sigmagluex[:,0]
Eavg_sigmagluex  = Ebeam_sigmagluex
sigma_sigmagluex, error_sigmagluex =  sigmagluex[:,1], sigmagluex[:,2]
Emin_sigmagluex,  Emax_sigmagluex =  sigmagluex[:,3],  sigmagluex[:,4]
t_sigmagluex = np.array([0. for i in range(len(Ebeam_sigmagluex))])
tmin_sigmagluex, tmax_sigmagluex = t_sigmagluex, t_sigmagluex
class_sigmagluex = np.array([0 for i in range(len(Ebeam_sigmagluex))])

file_sigmasolidp = "solid-1d-photoproduction.csv";
sigmasolidp = np.genfromtxt(file_sigmasolidp, delimiter=",");
ibeam_sigmasolidp = sigmasolidp[:,0]
Ebeam_sigmasolidp = sigmasolidp[:,1]
relsigma_sigmasolidp = sigmasolidp[:,2]

file_sigmasolide = "solid-1d-electroproduction.csv";
sigmasolide = np.genfromtxt(file_sigmasolide, delimiter=",");
ibeam_sigmasolide = sigmasolide[:,0]
Ebeam_sigmasolide = sigmasolide[:,1]
relsigma_sigmasolide = sigmasolide[:,2]

Datainput_gluexXsec = Namespace(clase=class_sigmagluex, eavg=Eavg_sigmagluex,
                            ebeam=Ebeam_sigmagluex, emin=Emin_sigmagluex, emax=Emax_sigmagluex,
                            t=t_sigmagluex, tmin=tmin_sigmagluex, tmax=tmax_sigmagluex,
                            obs=sigma_sigmagluex, error=error_sigmagluex)
npoints_gluexXsec = len(Datainput_gluexXsec.clase)

normalization_gluex_input = 0.20
normalization_gluex = np.array([normalization_gluex_input])
class_gluex = class_sigmagluex
Ebeam_gluex = Ebeam_sigmagluex
Eavg_gluex = Eavg_sigmagluex
obs_gluex = sigma_sigmagluex
error_gluex = error_sigmagluex
t_gluex = t_sigmagluex
tmin_gluex = tmin_sigmagluex
tmax_gluex = tmax_sigmagluex
Emin_gluex = Emin_sigmagluex
Emax_gluex = Emax_sigmagluex

Datainput_gluex = Namespace(clase=class_gluex, eavg=Eavg_gluex,
                            ebeam=Ebeam_gluex, emin=Emin_gluex, emax=Emax_gluex,
                            t=t_gluex, tmin=tmin_gluex, tmax=tmax_gluex,
                            obs=obs_gluex, error=error_gluex)
npoints_gluex = len(Datainput_gluex.clase)

Datainput = Datainput_gluex;
Normalization = normalization_gluex 

Data = Datainput
nclass = [np.count_nonzero(Datainput.clase==0), np.count_nonzero(Datainput.clase==1)]
ndata = np.sum(nclass)

###############################################################################
#   Files
###############################################################################

xsec_file_3cnr = np.loadtxt('plot_xsec_gluex_3cnr.txt')
xsec_file_3cr  = np.loadtxt('plot_xsec_gluex_3cr.txt')
xsec_file_1c   = np.loadtxt('plot_xsec_gluex_1c.txt')

###############################################################################
#   Calculations
###############################################################################

thresholds_E = [Ebeamfroms((mproton+mpsi)**2,mproton),Ebeamfroms((md+mlambdac)**2,mproton),Ebeamfroms((mdbar+mlambdac)**2,mproton)]
polefound = [8.522336073637502 ,8.966201905550529,8.966204511164975]

nplotpoints = 100
fuente = 20; 

sth = (mproton + mpsi + 0.001)**2
send = sfromEbeam(12.,mproton)
sarray = np.linspace(sth,send,nplotpoints)
Earray = Ebeamfroms(sarray,mproton)
xerror = (Emax_sigmagluex-Emin_sigmagluex)/2.

Earray, sarray, tarray, xsec, xsec_dw68, xsec_up68, xsec_dw95, xsec_up95 = xsec_file_3cr[0,:], xsec_file_3cr[1,:], xsec_file_3cr[2,:], xsec_file_3cr[3,:], xsec_file_3cr[4,:], xsec_file_3cr[5,:], xsec_file_3cr[6,:], xsec_file_3cr[7,:]
E3c, s3c, xdw683c, xup683c = Earray, sarray, xsec_dw68, xsec_up68
x3c = (xdw683c+xup683c)/2.

Earray, sarray, tarray, xsec, xsec_dw68, xsec_up68, xsec_dw95, xsec_up95 = xsec_file_3cnr[0,:], xsec_file_3cnr[1,:], xsec_file_3cnr[2,:], xsec_file_3cnr[3,:], xsec_file_3cnr[4,:], xsec_file_3cnr[5,:], xsec_file_3cnr[6,:], xsec_file_3cnr[7,:]
E3nc, s3nc, xdw683nc, xup683nc = Earray, sarray, xsec_dw68, xsec_up68
x3nc = (xdw683nc+xup683nc)/2.

Earray, sarray, tarray, xsec, xsec_dw68, xsec_up68, xsec_dw95, xsec_up95 = xsec_file_1c[0,:], xsec_file_1c[1,:], xsec_file_1c[2,:], xsec_file_1c[3,:], xsec_file_1c[4,:], xsec_file_1c[5,:], xsec_file_1c[6,:], xsec_file_1c[7,:]
E1c, s1c, xdw681c, xup681c = Earray, sarray, xsec_dw68, xsec_up68
x1c = (xdw681c+xup681c)/2.

sigma1c_inter = interpolation(E1c,x1c)

initial_tamano = len(Ebeam_sigmagluex) - 11
tamano = initial_tamano*2 
Ebeam_new = np.zeros(tamano)
sigma_new = np.ones(tamano)/4.
error_y = np.zeros(tamano)
j = 0
for i in range(initial_tamano):
    Ebeam_new[j]   = Ebeam_sigmagluex[i] - xerror[i]/2.
    error_y[j] = error_sigmagluex[i]/2.
    j = j + 1
    Ebeam_new[j] = Ebeam_sigmagluex[i] + xerror[i]/2.
    error_y[j] = error_sigmagluex[i]/2.
    j = j + 1

sigma1c = sigma1c_inter(Ebeam_new)
error_x = np.ones(len(sigma1c))*xerror[0]/2.

# SOLID points and errors
sigma1c_solidp = sigma1c_inter(Ebeam_sigmasolidp) * 1.15
sigma1c_solide = sigma1c_inter(Ebeam_sigmasolide)
for i in range(len(sigma1c_solidp)):
    relsigma_sigmasolidp[i] = relsigma_sigmasolidp[i]*sigma1c_solidp[i]
for i in range(len(sigma1c_solide)):
    relsigma_sigmasolide[i] = relsigma_sigmasolide[i]*sigma1c_solide[i]

###############################################################################
#   Plotting
###############################################################################

fig = plt.figure(figsize=(7,5))

plt.xlim((8.1,9.5))
plt.ylim((0.01,1.2))
plt.yscale("log")
plt.xlabel(r'$E_\gamma$ [GeV]',fontsize=15)
plt.ylabel(r'$\sigma (\gamma p \to J/\psi p)$ [nb]',fontsize=15)

# GlueX points
plt.errorbar(Ebeam_sigmagluex, sigma_sigmagluex, xerr=xerror, yerr=error_sigmagluex, fmt="o", markersize=3, capsize=5., c=jpac_color[9], alpha=0.5, zorder=1 ,label='GlueX (2023)')
plt.errorbar(Ebeam_new, sigma1c, yerr=error_y, fmt="s", markersize=4, capsize=0., c=jpac_color[10], alpha=1.0, zorder=3,label='GlueX Projection')

# SOLID projections
plt.errorbar(Ebeam_sigmasolidp, sigma1c_solidp, yerr=relsigma_sigmasolidp, fmt="o", markersize=4, capsize=0., c=jpac_color[4], alpha=1.0,zorder=3,label='SOLID Photoproduction Projection')
plt.errorbar(Ebeam_sigmasolide, sigma1c_solide, yerr=relsigma_sigmasolide, fmt="o", markersize=4, capsize=0., c=jpac_color[1], alpha=1.0,zorder=3,label='SOLID Electroproduction Projection')

plt.plot(E1c,x1c,'-',lw=2,c=jpac_color[0],alpha=1,zorder=1,label='Nonresonant')
plt.plot(E3c,x3c,'-',lw=2,c=jpac_color[3],alpha=1,zorder=2,label='Resonant')
#plt.plot(E3nc,x3nc,'-',lw=2,c=jpac_color[2],alpha=1,zorder=1,label='3C Nonresonant')
#plt.fill_between(E1c, xdw681c, xup681c, facecolor=jpac_color[0], interpolate=True, alpha=0.4,zorder=2)
#plt.fill_between(E3c, xdw683c, xup683c, facecolor=jpac_color[3], interpolate=True, alpha=0.4,zorder=2)
#plt.fill_between(E3nc, xdw683nc, xup683nc, facecolor=jpac_color[2], interpolate=True, alpha=0.4,zorder=2)

plt.vlines(thresholds_E[1],0.0,1.2,linestyles='dashed',colors=jpac_color[1],zorder=0,alpha=0.5)
plt.vlines(thresholds_E[2],0.0,1.2,linestyles='dashed',colors=jpac_color[1],zorder=0,alpha=0.5)
plt.text(8.52,0.85,r'$\bar{D}\Lambda_c$ thr.',fontsize=14,c=jpac_color[1])
plt.text(9.12,0.85,r'$\bar{D}^*\Lambda_c$ thr.',fontsize=14,c=jpac_color[1])
plt.tick_params(direction='in',labelsize=14)
#plt.legend(loc='lower right',ncol=2,frameon=True,fontsize=10)
plt.legend(loc='lower right',ncol=1,frameon=True,fontsize=12)

plt.show()
fig.savefig('plotbsgluex.pdf', bbox_inches='tight')
fig.savefig('plotbsgluex.png', bbox_inches='tight')
        
###############################################################################
###############################################################################
#
#   End of code
#
###############################################################################
###############################################################################



