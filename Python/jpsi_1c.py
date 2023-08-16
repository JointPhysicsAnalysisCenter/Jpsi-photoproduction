#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 5, 2023

Single channel with effective range

@author: cesar
"""

###############################################################################
#   Libraries
###############################################################################

import sys
import numpy as np
import matplotlib.pyplot as plt
from argparse import Namespace
from iminuit import Minuit

###############################################################################
###############################################################################
#
#   INPUT
#
###############################################################################
###############################################################################
#
#   Execution options
#
#   option: see opciones
#   dataset: 'gluex', '007', 'combined'
#   nmc: number of bff or bs fits  
#   lmax: highest partial wave to include in scattering length approximation
#   leff: highest partial wave to include in effective range approximation


###############################################################################
#   Input
###############################################################################

opciones = ['read','fit','bs','plot','plotlog','plotbs','plotlogbs','total','totalbs']

if len(sys.argv)<6:
    print('Number of input parameters should be 6 or 7, input was ',len(sys.argv))
    print('Input was:',sys.argv)
    print('Example of execution command: $python PcPhotoproduction.py fit gluex 10 4',)
    sys.exit('Exiting due to error in input')

option  = sys.argv[1]
dataset = sys.argv[2]
nmc  = int(sys.argv[3])
lmax = int(sys.argv[4])
leff = int(sys.argv[5])

ninputs = len(sys.argv)
if ninputs==7: bffinput = sys.argv[6]
if leff>lmax or leff<-1: sys.exit('Leff i larger than Lmax')
    
###############################################################################
#   End of input
###############################################################################

###############################################################################
#   JPAC color style
###############################################################################

jpac_blue   = "#1F77B4"; jpac_red    = "#D61D28"; jpac_green  = "#2CA02C"; 
jpac_orange = "#FF7F0E"; jpac_purple = "#9467BD"; jpac_brown  = "#8C564B";
jpac_pink   = "#E377C2"; jpac_gold   = "#BCBD22"; jpac_aqua   = "#17BECF"; 
jpac_grey   = "#7F7F7F";

jpac_color = [jpac_blue, jpac_red, jpac_green, jpac_orange, jpac_purple,
              jpac_brown, jpac_pink, jpac_gold, jpac_aqua, jpac_grey, 'black' ];

dashes, jpac_axes = 10*'-', jpac_color[10];

###############################################################################
#   Physical constants
###############################################################################

hbarc2 = 1./(2.56819e-6);
mproton, mpsi, mphoton = 0.938272, 3.0969160, 0.;
md, mdbar, mlambdac = 1.86484, 2.00685, 2.28646; 

###############################################################################
#   Kinematics
###############################################################################

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

###############################################################################
#   Phase space
###############################################################################

def PhaseSpace(si,m1,m2):
    s = si + 1j*0.00000001
    st = (m1+m2)**2
    xi = 1 - st/s
    q2 = kallen(s,m1**2,m2**2)/s/4.;
    q = np.sqrt(q2)
    rho = 2.*q/np.sqrt(s)
    log0 = rho*np.log((xi-rho)/(xi+rho))
    log1 = xi*(m2-m1)/(m1+m2)*np.log(m2/m1)
    return -(log0 +log1)/np.pi/16./np.pi

###############################################################################
#   Legendre Polynomials
###############################################################################

def LegPol(l,x):
    if   l==0: return 1;
    elif l==1: return x;
    elif l==2: return ( 3.*x**2 - 1. )/2.;
    elif l==3: return ( 5.*x**3 - 3.*x )/2.;
    elif l==4: return ( 35.*x**4 - 30.*x**2 + 3. )/8.;
    elif l==5: return ( 63.*x**5 - 70.*x**3 + 15.*x )/8.;
    elif l==6: return ( 231.*x**6 - 315.*x**4 + 105.*x**2 - 5. )/16.;
    else: sys.exit('Wrong angular momentum')
    return 0;

###############################################################################
#   Amplitude
###############################################################################


def Tamp(s,l,m1,m2,m3,m4,nl,al,bl):
    q2 = kallen(s,m3**2,m4**2)/s/4.;
    K = (q2**l)*(al+bl*q2);
    return K/(1.+PhaseSpace(s,m3,m4)*K);

def Amp(s,l,m1,m2,m3,m4,nl,al,bl):
    p2, q2 = kallen(s,m1**2,m2**2)/s/4., kallen(s,m3**2,m4**2)/s/4.;
    p, q = np.sqrt(p2), np.sqrt(q2);
    K = (q2**l)*(al+bl*q2);
    N = nl*(p*q)**l
    return N/(1.+PhaseSpace(s,m3,m4)*K);

def BcalL(s,t,l,m1,m2,m3,m4,nl,al,bl):
    x = costhetafromt(s,t,m1,m2,m3,m4)
    Lpol = LegPol(l,x)
    return (2.*l+1.)*Lpol*Amp(s,l,m1,m2,m3,m4,nl,al,bl); 

def Bcal(s,t,m1,m2,m3,m4,nl,al,bl,lmax):
    return np.sum([ BcalL(s,t,l,m1,m2,m3,m4,nl[l],al[l],bl[l]) for l in range(lmax+1)])

def singleBcal(s,t,l,m1,m2,m3,m4,nl,al,bl):
    return BcalL(s,t,l,m1,m2,m3,m4,nl[l],al[l],bl[l])

###############################################################################
#   Observables
###############################################################################

def dsigmadt_sc(s,t,m1,m2,m3,m4,nl,al,bl,lmax):
    amplitude = Bcal(s,t,m1,m2,m3,m4,nl,al,bl,lmax)
    num = np.absolute(amplitude)**2
    den = 16.*np.pi*(s-m2**2)**2
    return hbarc2*num/den;

def single_dsigmadt_sc(s,t,m1,m2,m3,m4,nl,al,bl,l):
    amplitude = singleBcal(s,t,l,m1,m2,m3,m4,nl,al,bl)
    num = np.absolute(amplitude)**2
    den = 16.*np.pi*(s-m2**2)**2
    return hbarc2*num/den;

def sigma_sc(s,m1,m2,m3,m4,nl,al,bl,lmax):
    p, q = momentum(s,m1,m2), momentum(s,m3,m4)
    num = np.sum([ (2*l+1)*np.absolute(Amp(s,l,m1,m2,m3,m4,nl[l],al[l],bl[l]))**2 for l in range(lmax+1)])
    den = 16.*np.pi*p*s
    return hbarc2*num*q/den;

def single_sigma_sc(s,m1,m2,m3,m4,nl,al,b1l,l):
    p, q = momentum(s,m1,m2), momentum(s,m3,m4)
    num = (2*l+1)*np.absolute(Amp(s,l,m1,m2,m3,m4,nl[l],al[l],bl[l]))**2
    den = 16.*np.pi*p*s
    return hbarc2*num*q/den;

def observable_sc(s,t,m1,m2,m3,m4,nl,al,bl,lmax,clase):
    if   clase==0: return sigma_sc(s,m1,m2,m3,m4,nl,al,bl,lmax);
    elif clase==1: return dsigmadt_sc(s,t,m1,m2,m3,m4,nl,al,bl,lmax);
    else: sys.exit('Wrong class')
    return 0;

def sigma_tot(s,m1,m2,m3,m4,nl,al,bl,lmax):
    den = np.sqrt(kallen(s,m3**2,m4**2))
    num = np.sum([ (2*l+1)*np.imag(Tamp(s,l,m1,m2,m3,m4,nl[l],al[l],bl[l])) for l in range(lmax+1)])
    return hbarc2*num/den/1.0e6;

###############################################################################
#   BS observables
###############################################################################

def bs_sigma_sc(xbs,sarray,m1,m2,m3,m4):
    ns, nbs  = len(sarray), len(xbs[:,0])
    idown68, iup68 = int(np.trunc(0.16*nbs)),  int(np.trunc(0.84*nbs))
    idown95, iup95 = int(np.trunc(0.025*nbs)), int(np.trunc(0.975*nbs))    
    avg, dw68, up68, dw95, up95 = np.zeros(ns), np.zeros(ns), np.zeros(ns), np.zeros(ns), np.zeros(ns);
    for j in range(ns):
        s = sarray[j]
        xsec = np.zeros(nbs)
        for ibs in range(nbs):
            input0 = xbs[ibs,:]
            parameters_input = np.array([ input0[i] for i in range(2,len(input0))])
            nl, al, bl = np.array_split(parameters_input,3)
            lmax = len(nl)-1
            xsec[ibs] = sigma_sc(s,m1,m2,m3,m4,nl,al,bl,lmax)
        xsecsorted = np.sort(xsec)
        avg[j] = np.mean(xsecsorted)
        dw68[j], up68[j], dw95[j], up95[j] = xsecsorted[idown68], xsecsorted[iup68], xsecsorted[idown95], xsecsorted[iup95]
    return avg, dw68, up68, dw95, up95

def bs_dsigmadt_sc(xbs,s,tarray,m1,m2,m3,m4):
    nt, nbs  = len(tarray), len(xbs[:,0])
    idown68, iup68 = int(np.trunc(0.16*nbs)),  int(np.trunc(0.84*nbs))
    idown95, iup95 = int(np.trunc(0.025*nbs)), int(np.trunc(0.975*nbs))    
    avg, dw68, up68, dw95, up95 = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
    for k in range(nt):
        t = tarray[k]
        dsdt = np.zeros(nbs)
        for ibs in range(nbs):
            input0 = xbs[ibs,:]
            parameters_input = np.array([ input0[i] for i in range(2,len(input0))])
            nl, al, bl = np.array_split(parameters_input,3)
            lmax = len(nl)-1
            dsdt[ibs] = dsigmadt_sc(s,t,m1,m2,m3,m4,nl,al,bl,lmax)
        dsdtsorted = np.sort(dsdt)
        avg[k] = np.mean(dsdtsorted)
        dw68[k], up68[k], dw95[k], up95[k] = dsdtsorted[idown68], dsdtsorted[iup68], dsdtsorted[idown95], dsdtsorted[iup95]
    return avg, dw68, up68, dw95, up95

def bs_total(xbs,sarray,m1,m2,m3,m4):
    ns, nbs  = len(sarray), len(xbs[:,0])
    idown68, iup68 = int(np.trunc(0.16*nbs)),  int(np.trunc(0.84*nbs))
    idown95, iup95 = int(np.trunc(0.025*nbs)), int(np.trunc(0.975*nbs))    
    dw68, up68, dw95, up95 = np.zeros(ns), np.zeros(ns), np.zeros(ns), np.zeros(ns);
    for j in range(ns):
        s = sarray[j]
        xsec = np.zeros(nbs)
        for ibs in range(nbs):
            input0 = xbs[ibs,:]
            parameters_input = np.array([ input0[i] for i in range(2,len(input0))])
            nl, al, bl = np.array_split(parameters_input,3)
            lmax = len(nl)-1
            xsec[ibs] =  sigma_tot(s,m1,m2,m3,m4,nl,al,bl,lmax)
        xsecsorted = np.sort(xsec)
        dw68[j], up68[j], dw95[j], up95[j] = xsecsorted[idown68], xsecsorted[iup68], xsecsorted[idown95], xsecsorted[iup95]
    return (dw68+up68)/2., dw68, up68, dw95, up95
    
###############################################################################
#   Fitting routine for MINUIT
###############################################################################

def LSQ_sc(par):
    m1, m2, m3, m4 = mphoton, mproton, mpsi, mproton;
    nl, al, bl = np.array_split(par,3)
    lmax = len(al) - 1
    s, t = sfromEbeam(Data.ebeam,m2), Data.t
    clase = Data.clase    
    func = [ observable_sc(s[i],t[i],m1,m2,m3,m4,nl,al,bl,lmax,clase[i]) for i in range(len(Data.ebeam))]
    return np.sum(((Data.obs-func)**2)/(Data.error**2))

def pull_sc(par):
    m1, m2, m3, m4 = mphoton, mproton, mpsi, mproton;
    nl, al, bl = np.array_split(par,3)
    lmax = len(al) - 1
    s, t = sfromEbeam(Data.ebeam,m2), Data.t
    clase = Data.clase    
    func = [ observable_sc(s[i],t[i],m1,m2,m3,m4,nl,al,bl,lmax,clase[i]) for i in range(len(Data.ebeam))]
    return (Data.obs-func)/Data.error

###############################################################################
#   Bootstrap dataset
###############################################################################

def pseudodataset(ydata,y_error):
    pseudodata = [ np.random.normal(ydata[i],y_error[i]) for i in np.arange(y_error.size)]
    return pseudodata

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

#   GlueX cross section: Class 0
file_sigmagluex = "sigma_gluex.txt"; sigmagluex = np.loadtxt(file_sigmagluex);
Ebeam_sigmagluex = sigmagluex[:,0]
Eavg_sigmagluex  = Ebeam_sigmagluex
sigma_sigmagluex, error_sigmagluex =  sigmagluex[:,1], sigmagluex[:,2]
Emin_sigmagluex,  Emax_sigmagluex =  sigmagluex[:,3],  sigmagluex[:,4]
t_sigmagluex = np.array([0. for i in range(len(Ebeam_sigmagluex))])
tmin_sigmagluex, tmax_sigmagluex = t_sigmagluex, t_sigmagluex
class_sigmagluex = np.array([0 for i in range(len(Ebeam_sigmagluex))])

#   GlueX ds/dt: Class 1
file_dsdtgluex = "dsdt_gluex.txt"; dsdtgluex = np.loadtxt(file_dsdtgluex);
t_dsdtgluex = -dsdtgluex[:,0]
dsdt_dsdtgluex, error_dsdtgluex =  dsdtgluex[:,1],  dsdtgluex[:,2]
tmax_dsdtgluex,  tmin_dsdtgluex = -dsdtgluex[:,3], -dsdtgluex[:,4]
Ebeam_dsdtgluex, Eavg_dsdtgluex =  dsdtgluex[:,5],  dsdtgluex[:,6]
Emin_dsdtgluex,  Emax_dsdtgluex =  Ebeam_dsdtgluex,  Ebeam_dsdtgluex
class_dsdtgluex = np.array([1 for i in range(len(t_dsdtgluex))])
id_dsdtgluex = dsdtgluex[:,7]

#   Full GlueX dataset
normalization_gluex_input = 0.20
normalization_gluex = np.array([normalization_gluex_input])
class_gluex = np.concatenate((class_sigmagluex,class_dsdtgluex))
Ebeam_gluex = np.concatenate((Ebeam_sigmagluex,Ebeam_dsdtgluex))
Eavg_gluex = np.concatenate((Eavg_sigmagluex,Eavg_dsdtgluex))
obs_gluex = np.concatenate((sigma_sigmagluex,dsdt_dsdtgluex))
error_gluex = np.concatenate((error_sigmagluex,error_dsdtgluex))
t_gluex = np.concatenate((t_sigmagluex,t_dsdtgluex))
tmin_gluex = np.concatenate((tmin_sigmagluex,tmin_dsdtgluex))
tmax_gluex = np.concatenate((tmax_sigmagluex,tmax_dsdtgluex))
Emin_gluex = np.concatenate((Emin_sigmagluex,Emin_dsdtgluex))
Emax_gluex = np.concatenate((Emax_sigmagluex,Emax_dsdtgluex))

Datainput_gluex = Namespace(clase=class_gluex, eavg=Eavg_gluex,
                            ebeam=Ebeam_gluex, emin=Emin_gluex, emax=Emax_gluex,
                            t=t_gluex, tmin=tmin_gluex, tmax=tmax_gluex,
                            obs=obs_gluex, error=error_gluex)
npoints_gluex = len(Datainput_gluex.clase)

#   Full Hall C J/psi-007 : Class 1
normalization_007_input = 0.04
normalization_007 = np.array([normalization_007_input])
file_dsdt007 = "dsdt_jpsi007.csv"; dsdt007 = np.loadtxt(file_dsdt007,delimiter=",",skiprows=1);
E_idx007 = dsdt007[:,2]
t_dsdt007 = -dsdt007[:,10]
dsdt_dsdt007, error_dsdt007 =  dsdt007[:,11],  dsdt007[:,14]
tmin_dsdt007, tmax_dsdt007 = -dsdt007[:,7], -dsdt007[:,6]
Eavg_dsdt007  = dsdt007[:,8]
Ebeam_dsdt007 = Eavg_dsdt007
Emin_dsdt007, Emax_dsdt007 =  dsdt007[:,4],  dsdt007[:,5]
class_dsdt007 = np.array([1 for i in range(len(t_dsdt007))])

Datainput_007 = Namespace(clase=class_dsdt007, eavg=Eavg_dsdt007,
                            ebeam=Ebeam_dsdt007, emin=Emin_dsdt007, emax=Emax_dsdt007,
                            t=t_dsdt007, tmin=tmin_dsdt007, tmax=tmax_dsdt007,
                            obs=dsdt_dsdt007, error=error_dsdt007)
npoints_007 = len(Datainput_007.clase)

#   Combined GlueX+J/psi-007 dataset
normalization_comb = np.array([normalization_gluex_input, normalization_007_input]);
class_comb = np.concatenate((class_gluex,class_dsdt007))
Ebeam_comb = np.concatenate((Ebeam_gluex,Ebeam_dsdt007))
Eavg_comb = np.concatenate((Eavg_gluex,Eavg_dsdt007))
obs_comb = np.concatenate((obs_gluex,dsdt_dsdt007))
error_comb = np.concatenate((error_gluex,error_dsdt007))
t_comb = np.concatenate((t_gluex,t_dsdt007))
tmin_comb = np.concatenate((tmin_gluex,tmin_dsdt007))
tmax_comb = np.concatenate((tmax_gluex,tmax_dsdt007))
Emin_comb = np.concatenate((Emin_gluex,Emin_dsdt007))
Emax_comb = np.concatenate((Emax_gluex,Emax_dsdt007))

Datainput_comb = Namespace(clase=class_comb, eavg=Eavg_comb,
                            ebeam=Ebeam_comb, emin=Emin_comb, emax=Emax_comb,
                            t=t_comb, tmin=tmin_comb, tmax=tmax_comb,
                            obs=obs_comb, error=error_comb)
npoints_comb = len(Datainput_gluex.clase) + len(Datainput_007.clase)

#   Dataset to fit
if   dataset == 'gluex':    
    Datainput = Datainput_gluex;
    Normalization = normalization_gluex 
elif dataset == '007':      
    Datainput = Datainput_007;
elif dataset == 'combined': 
    Datainput = Datainput_comb;
    Normalization = normalization_comb    
else: sys.exit('Wrong dataset')

Data = Datainput
nclass = [np.count_nonzero(Datainput.clase==0), np.count_nonzero(Datainput.clase==1)]
ndata = np.sum(nclass)
npar = 2*(lmax+1) + leff + 1

if option in opciones:
    print('Calculation:',option)
    print('Dataset:',dataset)
    print('Number of datapoints:',ndata)
    print('Number of datapoints per class:',nclass)
    print('Lmax:',lmax)
    print('Number of parameters:',npar)
else:
    sys.exit('Wrong option')    

#   Naming of parameters
vacio = []
for i in range(lmax+1):
    vacio.append('n'+str(i))
for i in range(lmax+1):
    vacio.append('a'+str(i))
for i in range(lmax+1):
    vacio.append('b'+str(i))
nombre = tuple( vacio[i] for i in range(len(vacio)) )

###############################################################################
#   Fitting. Exploring parameter space
###############################################################################

if option=='read' and ninputs==7:
    bff    = np.loadtxt(bffinput)
    input0 = bff[0,:]
    parameters_input = np.array([ input0[i] for i in range(2,len(input0))])
    nl, al, bl = np.array_split(parameters_input,3)
    lmax = len(nl)-1
    nnl, nal, nbl = lmax+1, lmax+1, lmax+1
    print('Lmax=',lmax)

    vacio = []
    for i in range(lmax+1): vacio.append('n0'+str(i))
    for i in range(lmax+1): vacio.append('a0'+str(i))
    for i in range(lmax+1): vacio.append('b0'+str(i))
    nombre = tuple( vacio[i] for i in range(len(vacio)) )

    for i in range(len(vacio)):
        print(vacio[i]+'=',input0[i+2])

    chi2 = LSQ_sc(parameters_input)
    print('chi2=',chi2,'chi2/N=',chi2/ndata)
    pull = pull_sc(parameters_input)
    print('Pull:')
    print(pull)
    print('Average pull=',np.mean(pull),'; Standard deviation=',np.std(pull))

elif option=='fit':

    #   Initialization of model parameters
    nnl, nal, nbl = lmax+1, lmax+1, lmax+1
    nlinput, alinput, blinput = [], [], [];
    
    if ninputs==7:
        bffinput = np.loadtxt(bffinput)
        input0 = bffinput[0,:]
        parameters_input = np.array([ input0[i] for i in range(2,len(input0))])
        nlinput_tmp, alinput_tmp, blinput_tmp = np.array_split(parameters_input,3)
        print('Initial parameters:',parameters_input )
        for i in range(nmc):
            len_tmp = len(nlinput_tmp)
            if len_tmp==(lmax+1):
                nlinput.append(nlinput_tmp)
                alinput.append(alinput_tmp)
                blinput.append(blinput_tmp)
            else:
                tmp = np.concatenate((nlinput_tmp,[0. for k in range(len_tmp,lmax+1)]),axis=0)
                nlinput.append(tmp)
                tmp = np.concatenate((alinput_tmp,[0. for k in range(len_tmp,lmax+1)]),axis=0)
                alinput.append(tmp)            
                tmp = np.concatenate((np.random.uniform(-5.,5.,leff+1),[0. for k in range(leff+1,nnl)]),axis=0)
                blinput.append(tmp)
    else:
        for i in range(nmc):
            if i==0:
                nlinput.append(np.random.uniform(0.,5.,nbl))
            else:
                nlinput.append(np.random.uniform(-5.,5.,nbl))
            alinput.append(np.random.uniform(-5.,5.,nal))
            tmp = np.concatenate((np.random.uniform(-5.,5.,leff+1),[0. for k in range(leff+1,nnl)]),axis=0)
            blinput.append(tmp)
            
    #   Fitting using MINUIT
    storage = []
    for i in range(nmc):
        #if i%10==0: print(i/nmc*100,'%')
        nlmc = np.array(nlinput[i])
        almc = np.array(alinput[i])  
        blmc = np.array(blinput[i])
        parameters_input = np.concatenate((nlmc,almc,blmc),axis=0)
        m_pc = Minuit(LSQ_sc,parameters_input,name=nombre)
        m_pc.errordef = Minuit.LEAST_SQUARES
        for il in range(leff+1,nnl):        
            nombre1 = 'b'+str(il)
            m_pc.fixed[nombre.index(nombre1)] = True
            
        m_pc.migrad(); #m_pc.hesse();
        chi2 = m_pc.fval
        chi2dof = chi2/(len(Datainput.obs)-npar)
        print(i+1,'out of',nmc,'chi2=',chi2,'chi2/dof=',chi2dof); 
    #    print(dashes); print(dashes);
        print(m_pc.params); 
    #    print(m_pc.covariance); print(m_pc.covariance.correlation())
        nl, al, bl = np.array_split(m_pc.values,3)
        storage.append( (chi2,chi2dof,nl,al,bl) )
    
    #   Sorting
    sorted_storage = sorted(storage, key=lambda chi2: chi2[0])
    
    #   File storage
    x_storage = []
    for i in range(nmc):
        x0, x1 = sorted_storage[i][0], sorted_storage[i][1]
        x2, x3 = sorted_storage[i][2][:], sorted_storage[i][3][:]
        x4 = sorted_storage[i][4][:]
        y0, y1 = [x0,x1], np.concatenate((x2,x3,x4),axis=0)
        if i==0: ybest = np.concatenate((x2,x3,x4),axis=0)
        x = np.concatenate((y0,y1),axis=0)
        x_storage.append(x)
    
    np.savetxt('pcbff.txt', x_storage)  

###############################################################################
#   Bootstrap
###############################################################################

elif option=='bs':
    
    #   Naming and fixing
    vacio = [];
    for i in range(lmax+1): 
        nome = 'n0'+str(i)
        vacio.append(nome);
    for i in range(lmax+1): 
        nome = 'a'+str(i)
        vacio.append(nome);
    for i in range(lmax+1): 
        nome = 'b'+str(i)
        vacio.append(nome);

    fixated = [ 0, 0, 0, 0, #   n0 n1 n2 n3
                0, 0, 0, 0, #   a0 a1 a2 a3
                0, 1, 1, 1, #   b0 b1 b2 b3
               ]

    #   Number of free parameters
    npar = len(fixated)-np.sum(np.array(fixated))
    nbs = nmc
    
    #   Initial values for the parameters
    if ninputs==7:
        bff    = np.loadtxt(bffinput)
    else:
        bff = np.loadtxt('pcbff.txt')

    input0 = bff[0,:]
    parameters_input = np.array([ input0[i] for i in range(2,len(input0))])
    nl, al, bl = np.array_split(parameters_input,3)
    lmax = len(nl)-1
    nnl, nal, nbl = lmax+1, lmax+1, lmax+1
    
    #   Pseudodata
    ypseudodata, output = [], [];
    output.append(Data.ebeam)
    output.append(Data.t)
    output.append(Data.error)
    for i in range(nbs):
        o = pseudodataset(Datainput.obs,Datainput.error)
        norm_gluex = np.random.normal(1,normalization_gluex_input)
        norm_007   = np.random.normal(1,normalization_007_input)
        if dataset=='gluex':
            on = [ norm_gluex*o[j] for j in range(npoints_gluex) ]
        elif dataset=='007':
            on = [ norm_007*o[j] for j in range(npoints_007) ]
        elif dataset=='combined':
            on = [None]*(npoints_gluex+npoints_007)
            k = 0
            for j in range(npoints_gluex):
                on[k] = norm_gluex*o[k]
                k = k + 1
            for j in range(npoints_007):        
                on[k] = norm_007*o[k]
                k = k + 1
        ypseudodata.append(on); output.append(on);
    np.savetxt('bsdata.txt', output);

    #   BS fits
    storage_bs = []
    for i in range(nbs):
        #if i%10==0: print(i/nmc*100,'%')
        Data.obs = np.array(ypseudodata[i])
        m_bs = Minuit(LSQ_sc,parameters_input)
        m_bs.errordef = Minuit.LEAST_SQUARES
        for kfix in range(len(fixated)): 
            if fixated[kfix]==1: m_bs.fixed[kfix] = True
        m_bs.migrad();
        chi2, chi2dof = m_bs.fval, m_bs.fval/(len(Datainput.obs)-npar);
        nl, al, bl = np.array_split(m_bs.values,3)

        storage_bs.append( (chi2,chi2dof,nl,al,bl) )

    #   Sorting bs fits
    sorted_storage_bs = sorted(storage_bs, key=lambda chi2: chi2[0])
    
    #   BS storage
    x_storage = []
    for i in range(nbs):
        x0, x1 = sorted_storage_bs[i][0], sorted_storage_bs[i][1]
        x2, x3 = sorted_storage_bs[i][2][:], sorted_storage_bs[i][3][:]
        x4 = sorted_storage_bs[i][4][:]
        y0, y1 = [x0,x1], np.concatenate((x2,x3,x4),axis=0)
        x = np.concatenate((y0,y1),axis=0)
        x_storage.append(x)
    np.savetxt('pcbs.txt', x_storage)
    
###############################################################################
#   Plot
###############################################################################
    
elif option=='plot' or option=='plotlog':

    fuente = 20; 
    nini, nfin = nmc, lmax

    if ninputs==7:
        bff = np.loadtxt(bffinput)
    else:
        bff = np.loadtxt('pcbff.txt')

    nfits = len(bff[:,0])

    sth = (mproton + mpsi + 0.0000001)**2
    send = sfromEbeam(12.,mproton)
    sarray = np.linspace(sth,send,100)
    Earray = Ebeamfroms(sarray,mproton)

    if dataset in ['gluex','combined']:
        
        xplots, yplots = 2, 2; 
        fig, subfig = plt.subplots(xplots,yplots,figsize=(15,15))
        xerror = (Emax_sigmagluex-Emin_sigmagluex)/2.
        
        if option=='plotlog':
            subfig[0,0].set_yscale('log')
            subfig[0,1].set_yscale('log')
            subfig[1,0].set_yscale('log')
            subfig[1,1].set_yscale('log')
            subfig[0,0].set_ylim((1e-3,1e1))
            subfig[0,1].set_ylim((1e-4,2e0))
            subfig[1,0].set_ylim((1e-4,2e0))
            subfig[1,1].set_ylim((1e-4,2e0))

        subfig[0,0].set_xlim((8,12))
        subfig[0,1].set_xlim((0.,10.))
        subfig[1,0].set_xlim((0.,10.))
        subfig[1,1].set_xlim((0.,10.))

        subfig[0,0].errorbar(Ebeam_sigmagluex, sigma_sigmagluex, xerr=xerror, yerr=error_sigmagluex, fmt="o", markersize=3,capsize=5., c=jpac_color[9], alpha=1,zorder=3)
        for ifit in range(nini,nfin):
            input0 = bff[ifit,:]
            parameters_input = np.array([ input0[i] for i in range(2,len(input0))])
            nl, al, bl = np.array_split(parameters_input,3)
            lmax = len(nl)-1
            xsec = [ sigma_sc(sarray[i],mphoton,mproton,mpsi,mproton,nl,al,bl,lmax) for i in range(len(sarray))]
            subfig[0,0].plot(Earray,xsec,'-',lw=2,c=jpac_color[0],alpha=1,zorder=2,label=r'$L_{max}$='+str(lmax))

            for l in range(lmax+1):
                pw2  = [ single_sigma_sc(sarray[i],mphoton,mproton,mpsi,mproton,nl,al,bl,l) for i in range(len(sarray))]
                subfig[0,0].plot(Earray,pw2,'--',lw=1,c=jpac_color[l],alpha=1,zorder=2,label=r'$\ell$='+str(l))
            
            for i in [0,1,2]:
                x, y, xerror1, xerror2, yerror = [], [], [], [], [];
                for j in range(len(id_dsdtgluex)):
                    if i==id_dsdtgluex[j]:
                        x.append(t_dsdtgluex[j])                        
                        y.append(dsdt_dsdtgluex[j])
                        xerror1.append(np.absolute(tmax_dsdtgluex[j]-t_dsdtgluex[j]))
                        xerror2.append(np.absolute(tmin_dsdtgluex[j]-t_dsdtgluex[j]))
                        yerror.append(error_dsdtgluex[j])
                        ebeam = Eavg_dsdtgluex[j]
                x, y, xerror1, xerror2, yerror = np.array(x), np.array(y), np.array(xerror1), np.array(xerror2), np.array(yerror)
                xerror = [xerror1,xerror2]
                savg = sfromEbeam(ebeam, mproton)
                tdw = tfromcostheta(savg,1.,mphoton,mproton,mpsi,mproton)
                tup = tfromcostheta(savg,-1.,mphoton,mproton,mpsi,mproton)
                tarray = np.linspace(tup,tdw,100)

                dsdt = [ dsigmadt_sc(savg,tarray[k],mphoton,mproton,mpsi,mproton,nl,al,bl,lmax) for k in range(len(tarray))]
                if i==0: 
                    subfig[0,1].errorbar(-x, y, xerr=xerror, yerr=yerror, fmt="o", markersize=3,capsize=5., c=jpac_color[9], alpha=1,zorder=3)
                    subfig[0,1].plot(-tarray,dsdt,'-',lw=2,c=jpac_color[0],alpha=1,zorder=2,label=r'$L_{max}$='+str(lmax))
                elif i==1:
                    subfig[1,0].errorbar(-x, y, xerr=xerror, yerr=yerror, fmt="o", markersize=3,capsize=5., c=jpac_color[9], alpha=1,zorder=3)
                    subfig[1,0].plot(-tarray,dsdt,'-',lw=2,c=jpac_color[0],alpha=1,zorder=2,label=r'$L_{max}$='+str(lmax))
                elif i==2:
                    subfig[1,1].errorbar(-x, y, xerr=xerror, yerr=yerror, fmt="o", markersize=3,capsize=5., c=jpac_color[9], alpha=1,zorder=3)
                    subfig[1,1].plot(-tarray,dsdt,'-',lw=2,c=jpac_color[0],alpha=1,zorder=2,label=r'$L_{max}$='+str(lmax))
                else:
                    sys.exit('Not a valid dataset')

                for l in range(lmax+1):
                    dsdt_pw = [ single_dsigmadt_sc(savg,tarray[k],mphoton,mproton,mpsi,mproton,nl,al,bl,l) for k in range(len(tarray))]
                    if i==0: 
                        subfig[0,1].plot(-tarray,dsdt_pw,'--',lw=1,c=jpac_color[l],alpha=1,zorder=2,label=r'$\ell$='+str(l))
                    elif i==1:
                        subfig[1,0].plot(-tarray,dsdt_pw,'--',lw=1,c=jpac_color[l],alpha=1,zorder=2,label=r'$\ell$='+str(l))
                    elif i==2:
                        subfig[1,1].plot(-tarray,dsdt_pw,'--',lw=1,c=jpac_color[l],alpha=1,zorder=2,label=r'$\ell$='+str(l))
                    else:
                        sys.exit('Not a valid dataset')

        subfig[0,0].set_xlabel(r'$E_\gamma$ (GeV)',fontsize=fuente)
        subfig[0,1].set_xlabel(r'$-t$ (GeV$^2$)',fontsize=fuente)
        subfig[1,0].set_xlabel(r'$-t$ (GeV$^2$)',fontsize=fuente)
        subfig[1,1].set_xlabel(r'$-t$ (GeV$^2$)',fontsize=fuente)

        subfig[0,0].set_ylabel(r'$\sigma (nb)$',fontsize=fuente)
        subfig[0,1].set_ylabel(r'$d\sigma/dt (nb/GeV^2)$',fontsize=fuente)
        subfig[1,0].set_ylabel(r'$d\sigma/dt (nb/GeV^2)$',fontsize=fuente)
        subfig[1,1].set_ylabel(r'$d\sigma/dt (nb/GeV^2)$',fontsize=fuente)

        subfig[0,0].tick_params(direction='in',labelsize=fuente)
        subfig[0,1].tick_params(direction='in',labelsize=fuente)
        subfig[1,0].tick_params(direction='in',labelsize=fuente)
        subfig[1,1].tick_params(direction='in',labelsize=fuente)
        
        if option=='plotlog':
            subfig[0,0].legend(loc='lower right',ncol=1,frameon=True,fontsize=11)
            subfig[0,1].legend(loc='upper right',ncol=1,frameon=True,fontsize=11)
            subfig[1,0].legend(loc='upper right',ncol=1,frameon=True,fontsize=11)
            subfig[1,1].legend(loc='upper right',ncol=1,frameon=True,fontsize=11)
        else:
            subfig[0,0].legend(loc='upper left',ncol=1,frameon=True,fontsize=11)
            subfig[0,1].legend(loc='upper right',ncol=1,frameon=True,fontsize=11)
            subfig[1,0].legend(loc='upper right',ncol=1,frameon=True,fontsize=11)
            subfig[1,1].legend(loc='upper right',ncol=1,frameon=True,fontsize=11)

        fig.savefig('plotgluex.pdf', bbox_inches='tight')

    if dataset=='007' or dataset=='combined':

        ista, iend = int(np.min(E_idx007)), int(np.max(E_idx007));
        xplots, yplots = int((iend-ista)/4+1), 4
        fig, subfig = plt.subplots(xplots,yplots,figsize=(7*yplots,7*xplots))
        idx = range(ista,iend+1)
        k = 0
        for i in range(xplots):
            for j in range(yplots):
                idxarray = np.where(E_idx007==idx[k])
                for ide in idxarray[0]:
                    x, y = -Datainput_007.t[ide], Datainput_007.obs[ide]
                    yerror = Datainput_007.error[ide]
                    ebeam_text = str(Datainput_007.ebeam[ide])
                    subfig[i,j].errorbar(x,y,yerr=yerror, fmt="o", markersize=3,capsize=5., c=jpac_color[9], alpha=1,zorder=3)

                ebeam = Datainput_007.eavg[ide]
                savg = sfromEbeam(ebeam, mproton)
                tdw = tfromcostheta(savg,1.,mphoton,mproton,mpsi,mproton)
                tup = tfromcostheta(savg,-1.,mphoton,mproton,mpsi,mproton)
                tarray = np.linspace(tup,tdw,100)
                for ifit in range(nini,nfin):
                    input0 = bff[ifit,:]
                    parameters_input = np.array([ input0[i] for i in range(2,len(input0))])
                    nl, al, bl = np.array_split(parameters_input,3)
                    lmax = len(nl)-1
                    dsdt = [ dsigmadt_sc(savg,tarray[k],mphoton,mproton,mpsi,mproton,nl,al,bl,l) for k in range(len(tarray))]
                    subfig[i,j].plot(-tarray,dsdt,'-',lw=2,c=jpac_color[0],alpha=1,zorder=2,label=r'$L_{max}$='+str(lmax))
                    for l in range(lmax+1):
                        dsdt_pw = [ single_dsigmadt_sc(savg,tarray[ki],mphoton,mproton,mpsi,mproton,nl,al,bl,l) for ki in range(len(tarray))]
                        subfig[i,j].plot(-tarray,dsdt_pw,'--',lw=1,c=jpac_color[l],alpha=1,zorder=2,label=r'$\ell$='+str(l))
                if i==(xplots-1): subfig[i,j].set_xlabel(r'$-t$ (GeV$^2$)',fontsize=fuente)
                if j==0: subfig[i,j].set_ylabel(r'$d\sigma/dt (nb/GeV^2)$',fontsize=fuente)
                subfig[i,j].set_xlim((0.,-tup))
                subfig[i,j].set_ylim((0.,1.5))
                if option=='plotlog': 
                    subfig[i,j].set_yscale('log')      
                    subfig[i,j].set_ylim((1e-3,1.5e0))
                subfig[i,j].tick_params(direction='in',labelsize=fuente)
                subfig[i,j].legend(loc='upper right',ncol=1,frameon=True,fontsize=11)
                k = k +1
        fig.savefig('plot007.pdf', bbox_inches='tight')
        
elif option=='plotbs' or option=='plotlogbs':

    nplotpoints = 100
    fuente = 20; 

    norm_gluex, norm_007 = 1.02, 1.04

    if nmc==0:
        if dataset=='gluex' or dataset=='combined':            
            xsec_file  = np.loadtxt('plot_xsec_gluex.txt')
            dsdt_file = []
            dsdt_file.append(np.loadtxt('plot_dsdt_gluex_0.txt'))
            dsdt_file.append(np.loadtxt('plot_dsdt_gluex_1.txt'))
            dsdt_file.append(np.loadtxt('plot_dsdt_gluex_2.txt'))
        if dataset=='007' or dataset=='combined':
            ista, iend = int(np.min(E_idx007)), int(np.max(E_idx007));
            xplots, yplots = int((iend-ista)/4+1), 4
            idx = range(ista,iend+1)
            k = 0
            dsdt_007_all = []
            for i in range(xplots):
                for j in range(yplots):
                    filestoragename = 'plot_dsdt_007'+str(k)+'.txt'
                    dsdt_007_all.append(np.loadtxt(filestoragename))
                    k = k +1
    else:
        if ninputs==7:
            bsf = np.loadtxt(bffinput)
        else:
            bsf = np.loadtxt('pcbs.txt')

        nfits = len(bsf[:,0])
        print('Number of BS fits=',nfits)
    
        sth = (mproton + mpsi + 0.001)**2
        send = sfromEbeam(12.,mproton)
        sarray = np.linspace(sth,send,nplotpoints)
        Earray = Ebeamfroms(sarray,mproton)
        
        storage_plot, storage_plot0 = np.zeros((8,nplotpoints)), np.zeros((8,nplotpoints))

    if dataset=='gluex' or dataset=='combined':

        xplots, yplots = 2, 2; 
        fig, subfig = plt.subplots(xplots,yplots,figsize=(15,15))
        xerror = (Emax_sigmagluex-Emin_sigmagluex)/2.

        if nmc==0:
            Earray, sarray, tarray, xsec, xsec_dw68, xsec_up68, xsec_dw95, xsec_up95 = xsec_file[0,:], xsec_file[1,:], xsec_file[2,:], xsec_file[3,:], xsec_file[4,:], xsec_file[5,:], xsec_file[6,:], xsec_file[7,:]
        else:            
            xsec, xsec_dw68, xsec_up68, xsec_dw95, xsec_up95 = bs_sigma_sc(bsf,sarray,mphoton,mproton,mpsi,mproton)
            storage_plot[0,:], storage_plot[1,:], storage_plot[2,:] = Earray, sarray, np.zeros(nplotpoints)
            storage_plot[3,:] = xsec
            storage_plot[4,:], storage_plot[5,:] = xsec_dw68, xsec_up68
            storage_plot[6,:], storage_plot[7,:] = xsec_dw95, xsec_up95
            np.savetxt('plot_xsec_gluex.txt', storage_plot)

        xsec = (xsec_up68 + xsec_dw68)/2.
        new_up68 = xsec_up68
        new_dw68 = xsec_dw68
        new_up95 = xsec_up95
        new_dw95 = xsec_dw95

        if option=='plotlogbs':
            subfig[0,0].set_yscale('log')
            subfig[0,1].set_yscale('log')
            subfig[1,0].set_yscale('log')
            subfig[1,1].set_yscale('log')

        subfig[0,0].set_xlim((8,12))
        subfig[0,1].set_xlim((0.,10.))
        subfig[1,0].set_xlim((0.,10.))
        subfig[1,1].set_xlim((0.,10.))

        subfig[0,0].errorbar(Ebeam_sigmagluex, sigma_sigmagluex, xerr=xerror, yerr=error_sigmagluex, fmt="o", markersize=3,capsize=5., c=jpac_color[10], alpha=1,zorder=1)
        subfig[0,0].plot(Earray,xsec,'-',lw=2,c=jpac_color[0],alpha=1,zorder=1)
        subfig[0,0].fill_between(Earray, new_dw68, new_up68, facecolor=jpac_color[0], interpolate=True, alpha=0.6,zorder=2)
        subfig[0,0].fill_between(Earray, new_dw68, new_dw95, facecolor=jpac_color[2], interpolate=True, alpha=0.3,zorder=3)
        subfig[0,0].fill_between(Earray, new_up68, new_up95, facecolor=jpac_color[2], interpolate=True, alpha=0.3,zorder=3)

        for i in [0,1,2]:
            x, y, xerror1, xerror2, yerror = [], [], [], [], [];
            for j in range(len(id_dsdtgluex)):
                if i==id_dsdtgluex[j]:
                    x.append(t_dsdtgluex[j])                        
                    y.append(dsdt_dsdtgluex[j])
                    xerror1.append(np.absolute(tmax_dsdtgluex[j]-t_dsdtgluex[j]))
                    xerror2.append(np.absolute(tmin_dsdtgluex[j]-t_dsdtgluex[j]))
                    yerror.append(error_dsdtgluex[j])
                    ebeam = Eavg_dsdtgluex[j]
            x, y, xerror1, xerror2, yerror = np.array(x), np.array(y), np.array(xerror1), np.array(xerror2), np.array(yerror)
            xerror = [xerror1,xerror2]
            savg = sfromEbeam(ebeam, mproton)
            tdw = tfromcostheta(savg,1.,mphoton,mproton,mpsi,mproton)
            tup = tfromcostheta(savg,-1.,mphoton,mproton,mpsi,mproton)
  
            if nmc==0:
                Earray, sarray, tarray, dsdt, dsdt_dw68, dsdt_up68, dsdt_dw95, dsdt_up95 = dsdt_file[i][0,:], dsdt_file[i][1,:], dsdt_file[i][2,:], dsdt_file[i][3,:], dsdt_file[i][4,:], dsdt_file[i][5,:], dsdt_file[i][6,:], dsdt_file[i][7,:]
            else:                        
                tarray = np.linspace(tup,tdw,100)
                dsdt, dsdt_dw68, dsdt_up68, dsdt_dw95, dsdt_up95 = bs_dsigmadt_sc(bsf,savg,tarray,mphoton,mproton,mpsi,mproton)
                storage_plot0[0,:], storage_plot0[1,:], storage_plot0[2,:] = np.full(nplotpoints,ebeam), np.full(nplotpoints,savg), tarray
                storage_plot0[3,:] = dsdt
                storage_plot0[4,:], storage_plot0[5,:] = dsdt_dw68, dsdt_up68
                storage_plot0[6,:], storage_plot0[7,:] = dsdt_dw95, dsdt_up95
    
            dsdt = (dsdt_up68 + dsdt_dw68)/2.
            new_up68 = dsdt_up68
            new_dw68 = dsdt_dw68
            new_up95 = dsdt_up95 
            new_dw95 = dsdt_dw95

            if i==0: 
                subfig[0,1].errorbar(-x, y, xerr=xerror, yerr=yerror, fmt="o", markersize=3,capsize=5., c=jpac_color[10], alpha=1,zorder=1)
                subfig[0,1].plot(-tarray,dsdt,'-',lw=2,c=jpac_color[0],alpha=1,zorder=1)
                subfig[0,1].fill_between(-tarray, new_dw68, new_up68, facecolor=jpac_color[0], interpolate=True, alpha=0.6,zorder=1)
                subfig[0,1].fill_between(-tarray, new_dw68, new_dw95, facecolor=jpac_color[2], interpolate=True, alpha=0.3,zorder=1)
                subfig[0,1].fill_between(-tarray, new_up68, new_up95, facecolor=jpac_color[2], interpolate=True, alpha=0.3,zorder=1)

                if nmc!=0:
                    np.savetxt('plot_dsdt_gluex_0.txt', storage_plot0)
                    print('first dsdt computed and stored')

            elif i==1:
                subfig[1,0].errorbar(-x, y, xerr=xerror, yerr=yerror, fmt="o", markersize=3,capsize=5., c=jpac_color[10], alpha=1,zorder=1)
                subfig[1,0].plot(-tarray,dsdt,'-',lw=2,c=jpac_color[0],alpha=1,zorder=1)
                subfig[1,0].fill_between(-tarray, new_dw68, new_up68, facecolor=jpac_color[0], interpolate=True, alpha=0.6,zorder=1)
                subfig[1,0].fill_between(-tarray, new_dw68, new_dw95, facecolor=jpac_color[2], interpolate=True, alpha=0.3,zorder=1)
                subfig[1,0].fill_between(-tarray, new_up68, new_up95, facecolor=jpac_color[2], interpolate=True, alpha=0.3,zorder=1)
                if nmc!=0:
                    np.savetxt('plot_dsdt_gluex_1.txt', storage_plot0)
                    print('second dsdt computed and stored')

            elif i==2:
                subfig[1,1].errorbar(-x, y, xerr=xerror, yerr=yerror, fmt="o", markersize=3,capsize=5., c=jpac_color[10], alpha=1,zorder=1)
                subfig[1,1].plot(-tarray,dsdt,'-',lw=2,c=jpac_color[0],alpha=1,zorder=1)
                subfig[1,1].fill_between(-tarray, new_dw68, new_up68, facecolor=jpac_color[0], interpolate=True, alpha=0.6,zorder=2)
                subfig[1,1].fill_between(-tarray, new_dw68, new_dw95, facecolor=jpac_color[2], interpolate=True, alpha=0.3,zorder=3)
                subfig[1,1].fill_between(-tarray, new_up68, new_up95, facecolor=jpac_color[2], interpolate=True, alpha=0.3,zorder=3)
                if nmc!=0:
                    np.savetxt('plot_dsdt_gluex_2.txt', storage_plot0)
                    print('third dsdt computed and stored')
            else:
                sys.exit('Not a valid dataset')

        subfig[0,0].set_xlabel(r'$E_\gamma$ (GeV)',fontsize=fuente)
        subfig[0,1].set_xlabel(r'$-t$ (GeV$^2$)',fontsize=fuente)
        subfig[1,0].set_xlabel(r'$-t$ (GeV$^2$)',fontsize=fuente)
        subfig[1,1].set_xlabel(r'$-t$ (GeV$^2$)',fontsize=fuente)

        subfig[0,0].set_ylabel(r'$\sigma (nb)$',fontsize=fuente)
        subfig[0,1].set_ylabel(r'$d\sigma/dt (nb/GeV^2)$',fontsize=fuente)
        subfig[1,0].set_ylabel(r'$d\sigma/dt (nb/GeV^2)$',fontsize=fuente)
        subfig[1,1].set_ylabel(r'$d\sigma/dt (nb/GeV^2)$',fontsize=fuente)

        subfig[0,0].tick_params(direction='in',labelsize=fuente)
        subfig[0,1].tick_params(direction='in',labelsize=fuente)
        subfig[1,0].tick_params(direction='in',labelsize=fuente)
        subfig[1,1].tick_params(direction='in',labelsize=fuente)
        
        fig.savefig('plotbsgluex.pdf', bbox_inches='tight')
        
    if dataset=='007' or dataset=='combined':

        storage_plot = np.zeros((8,nplotpoints))

        ista, iend = int(np.min(E_idx007)), int(np.max(E_idx007));
        xplots, yplots = int((iend-ista)/4+1), 4
        fig, subfig = plt.subplots(xplots,yplots,figsize=(7*yplots,7*xplots))
        idx = range(ista,iend+1)
        k = 0
        for i in range(xplots):
            for j in range(yplots):
                idxarray = np.where(E_idx007==idx[k])
                for ide in idxarray[0]:
                    x, y = -Datainput_007.t[ide], Datainput_007.obs[ide]
                    yerror = Datainput_007.error[ide]
                    ebeam_text = str(Datainput_007.ebeam[ide])
                    subfig[i,j].errorbar(x,y,yerr=yerror, fmt="o", markersize=3,capsize=5., c=jpac_color[10], alpha=1,zorder=3)

                ebeam = Datainput_007.eavg[ide]
                savg = sfromEbeam(ebeam, mproton)
                tdw = tfromcostheta(savg,1.,mphoton,mproton,mpsi,mproton)
                tup = tfromcostheta(savg,-1.,mphoton,mproton,mpsi,mproton)
                if nmc==0:
                    Earray, sarray, tarray, dsdt, dsdt_dw68, dsdt_up68, dsdt_dw95, dsdt_up95 = dsdt_007_all[k][0,:], dsdt_007_all[k][1,:], dsdt_007_all[k][2,:], dsdt_007_all[k][3,:], dsdt_007_all[k][4,:], dsdt_007_all[k][5,:], dsdt_007_all[k][6,:], dsdt_007_all[k][7,:]
                else:                
                    tarray = np.linspace(tup,tdw,100)
                    dsdt, dsdt_dw68, dsdt_up68, dsdt_dw95, dsdt_up95 = bs_dsigmadt_sc(bsf,savg,tarray,mphoton,mproton,mpsi,mproton)
                    storage_plot[0,:], storage_plot[1,:], storage_plot[2,:] = np.full(nplotpoints,ebeam), np.full(nplotpoints,savg), tarray
                    storage_plot[3,:] = dsdt
                    storage_plot[4,:], storage_plot[5,:] = dsdt_dw68, dsdt_up68
                    storage_plot[6,:], storage_plot[7,:] = dsdt_dw95, dsdt_up95
                    filestoragename = 'plot_dsdt_007'+str(k)+'.txt'
                    np.savetxt(filestoragename, storage_plot)
                    
                dsdt = (dsdt_up68 + dsdt_dw68)/2.
                new_up68 = dsdt_up68
                new_dw68 = dsdt_dw68
                new_up95 = dsdt_up95 
                new_dw95 = dsdt_dw95

                subfig[i,j].plot(-tarray,dsdt,'-',lw=2,c=jpac_color[0],alpha=1,zorder=1)
                subfig[i,j].fill_between(-tarray, new_dw68, new_up68, facecolor=jpac_color[0], interpolate=True, alpha=0.6,zorder=2)
                subfig[i,j].fill_between(-tarray, new_dw68, new_dw95, facecolor=jpac_color[2], interpolate=True, alpha=0.3,zorder=3)
                subfig[i,j].fill_between(-tarray, new_up68, new_up95, facecolor=jpac_color[2], interpolate=True, alpha=0.3,zorder=3)
                if i==(xplots-1): subfig[i,j].set_xlabel(r'$-t$ (GeV$^2$)',fontsize=fuente)
                if j==0: subfig[i,j].set_ylabel(r'$d\sigma/dt (nb/GeV^2)$',fontsize=fuente)
                subfig[i,j].set_xlim((0.,-tup))
                subfig[i,j].set_ylim((0.,1.5))
                if option=='plotlogbs': 
                    subfig[i,j].set_yscale('log')      
                    subfig[i,j].set_ylim((1e-3,1.5e0))
                subfig[i,j].tick_params(direction='in',labelsize=fuente)
                k = k +1
        fig.savefig('plotbs007.pdf', bbox_inches='tight')
        
elif option=='total':
    fuente = 20; 
    nini, nfin = nmc, lmax
    nplotpoints = 100

    if ninputs==7:
        bff = np.loadtxt(bffinput)
    else:
        bff = np.loadtxt('pcbff.txt')

    nfits = len(bff[:,0])

    sth = (mproton + mpsi + 0.0000001)**2
    send = sfromEbeam(15.,mproton)
    sarray = np.linspace(sth,send,nplotpoints)
    Earray = Ebeamfroms(sarray,mproton)
    
    storage_plot = np.zeros((3,nplotpoints))

    for ifit in range(nini,nfin):
        input0 = bff[ifit,:]
        parameters_input = np.array([ input0[i] for i in range(2,len(input0))])
        nl, al, bl = np.array_split(parameters_input,3)
        lmax = len(nl)-1
        xsec = [ sigma_tot(sarray[i],mphoton,mproton,mpsi,mproton,nl,al,bl,lmax) for i in range(len(sarray))]
        fig = plt.figure()
        plt.xlim((Ebeamfroms(sth,mproton),15));
        plt.yscale('log'); plt.ylim(10e-2, 10e3)
        plt.plot(Earray,xsec,'-',lw=2,c=jpac_color[0],alpha=1,zorder=2)
        fig.savefig('sigmatot.pdf', bbox_inches='tight')
        storage_plot[0,:], storage_plot[1,:], storage_plot[2,:] = Earray, sarray, xsec
        np.savetxt('sigmatot.txt', storage_plot)

elif option=='totalbs':

    nplotpoints = 100
    fuente = 20; 
    sth = (mproton + mpsi + 0.001)**2

    if nmc==0:
        xsec_file  = np.loadtxt('plot_totalbs.txt')
    else:
        if ninputs==7:
            bsf = np.loadtxt(bffinput)
        else:
            bsf = np.loadtxt('pcbs.txt')
        nfits = len(bsf[:,0])
        print('Number of BS fits=',nfits)
        send = sfromEbeam(15.,mproton)
        sarray = np.linspace(sth,send,nplotpoints)
        Earray = Ebeamfroms(sarray,mproton)
        
    storage_plot = np.zeros((8,nplotpoints))

    if nmc==0:
        Earray, sarray, tarray, xsec, xsec_dw68, xsec_up68, xsec_dw95, xsec_up95 = xsec_file[0,:], xsec_file[1,:], xsec_file[2,:], xsec_file[3,:], xsec_file[4,:], xsec_file[5,:], xsec_file[6,:], xsec_file[7,:]
    else:            
        xsec, xsec_dw68, xsec_up68, xsec_dw95, xsec_up95 = bs_total(bsf,sarray,mphoton,mproton,mpsi,mproton)
        storage_plot[0,:], storage_plot[1,:], storage_plot[2,:] = Earray, sarray, np.zeros(nplotpoints)
        storage_plot[3,:] = xsec
        storage_plot[4,:], storage_plot[5,:] = xsec_dw68, xsec_up68
        storage_plot[6,:], storage_plot[7,:] = xsec_dw95, xsec_up95
        np.savetxt('plot_totalbs.txt', storage_plot)

    new_up68 = xsec_up68
    new_dw68 = xsec_dw68
    new_up95 = xsec_up95
    new_dw95 = xsec_dw95

    fig = plt.figure()
    plt.xlim((Ebeamfroms(sth,mproton),15));
    plt.ylim(0, 60);

    plt.plot(Earray,xsec,'-',lw=2,c=jpac_color[0],alpha=1,zorder=1)
    plt.fill_between(Earray, new_dw68, new_up68, facecolor=jpac_color[0], interpolate=True, alpha=0.6,zorder=2)
    plt.fill_between(Earray, new_dw68, new_dw95, facecolor=jpac_color[2], interpolate=True, alpha=0.3,zorder=3)
    plt.fill_between(Earray, new_up68, new_up95, facecolor=jpac_color[2], interpolate=True, alpha=0.3,zorder=3)

    fig.savefig('sigmatotbs.pdf', bbox_inches='tight')
    fig.savefig('sigmatotbs.png', bbox_inches='tight')
        
else:
    sys.exit('Not a valid option')

###############################################################################
###############################################################################
#
#   End of code
#
###############################################################################
###############################################################################



