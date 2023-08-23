# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 15:39:19 2022

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
import copy

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
#   lmax: highest partial wave to include
#   model: see modelos

###############################################################################
#   Input
###############################################################################

opciones = ['fit','bs','plot','plotlog','plotbs','plotlogbs','test','polebff','polecheck','read','total','totalbs']
modelos  = ['init','sfree','scat2']

if len(sys.argv)<6:
    print('Number of input parameters should be 6 or 7, input was ',len(sys.argv))
    print('Input was:',sys.argv)
    print('Example of execution command: $python PcPhotoproduction.py fit gluex 10 4',)
    sys.exit('Exiting due to error in input')

option  = sys.argv[1]
dataset = sys.argv[2]
nmc     = int(sys.argv[3])
lmax    = int(sys.argv[4])
modelo  = sys.argv[5]

ninputs = len(sys.argv)
if ninputs==7: bffinput = sys.argv[6]

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

def momentum2(s,m1,m2):
    return kallen(s,m1**2,m2**2)/s/4.;

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

"""
    Finds the root of a complex-valued function using a variant of the regula falsi method.
    
    Parameters:
    func: The complex-valued function to evaluate.
    xa: The first initial guess for the root.
    xb: The second initial guess for the root.
    tol: The tolerance for the root, i.e. the maximum allowed difference between
         the magnitudes of the function's output at consecutive iterations.
         The default value is 1e-12.
    max_iter: The maximum number of iterations to perform. The default value is 16.
    
    Returns:
    The root of the function, or None if the root could not be found within the
    specified tolerance and number of iterations.
    """
def csearch(func,xa,xb,irs,m1,m2,m3,m4,a00,a01,a11,b00,b01,b11,l,tol=1e-12,max_iter=32):
    # Evaluating the function and its magnitude in the points xa and xb
    fa   = func(xa,irs,m1,m2,m3,m4,a00,a01,a11,b00,b01,b11,l)
    fa_m = np.abs(fa)
    fb   = func(xb,irs,m1,m2,m3,m4,a00,a01,a11,b00,b01,b11,l)
    fb_m = np.abs(fb)

    # Make sure xa is the point with the smaller magnitude of func
    if fa_m < fb_m:
        xc, fc = xa, fa
        xa, fa = xb, fb
        xb, fb = xc, fc

    # Iterate until the root is found or the maximum number of iterations is reached.
    for j in range(max_iter):  
        # question: bad convergence??? (if too slow, stop!)
        f0   = fa - fb
        f0_m = np.abs(f0)

        if f0_m <= tol:
            xa, fa = xb, fb
            #print('Bad convergence')
            return -1.+0.00000001*1j, 100.
            #raise TypeError('Bad convergence')

        # convergence is fast enough, we proceed with a linear approximation
        # to the complex function and guess the next argument.
        x0   = xb - fb * (xb - xa) / (fb - fa)
        f0   = func(x0,irs,m1,m2,m3,m4,a00,a01,a11,b00,b01,b11,l)
        f0_m = np.abs(f0)

        # the search continues. Reorder arguments.
        if f0_m < fb_m:
            xa, fa, fa_m = xb, fb, fb_m
            xb, fb, fb_m = x0, f0, f0_m
        else:
            xa, fa, fa_m = x0, f0, f0_m

        # Check if the root has been found
        if fb_m <= tol:
            xa, fa = xb, fb
            return xa, fb_m 
    
    # The root could not be found within the specified tolerance and number of iterations.
    #raise TypeError('In Riemann sheet',irs,', no solutions found')
    #warnings.warn('No solutions found')
    return -1.+0.00000001*1j, 100.
     
###############################################################################
#   Phase space
###############################################################################

def PhaseSpace(si,m1,m2):
    s  = si + 1j*0.00000001
    st = (m1+m2)**2
    xi = 1 - st/s
    q2 = kallen(s,m1**2,m2**2)/s/4.;
    q  = np.sqrt(q2)
    rho  = 2.*q/np.sqrt(s)
    log0 = rho*np.log((xi-rho)/(xi+rho))
    log1 = xi*(m2-m1)/(m1+m2)*np.log(m2/m1)
    return -(log0 +log1)/np.pi/16./np.pi

def PhaseSpaceI(s,m1,m2):
    st = (m1+m2)**2
    xi = 1 - st/s
    q2 = kallen(s,m1**2,m2**2)/s/4.;
    q  = np.sqrt(q2)
    rho  = 2.*q/np.sqrt(s)
    log0 = rho*np.log((xi-rho)/(xi+rho))
    log1 = xi*(m2-m1)/(m1+m2)*np.log(m2/m1)
    return -(log0 +log1)/np.pi/16./np.pi

def Denominator(s,irs,m1,m2,m3,m4,a00,a01,a11,b00,b01,b11,l):
    q02, q12 = kallen(s,m1**2,m2**2)/s/4., kallen(s,m3**2,m4**2)/s/4.;
    q0, q1 = np.sqrt(q02), np.sqrt(q12);
    K00  = (q02**l)*( a00 + b00*q02 )
    K11  = (q12**l)*( a11 + b11*q12 )
    K012 = ((q02*q12)**l)*( a01*a01 + b01*b01 + 2.*a01*b01*q0*q1 )
    G0, G1   = PhaseSpaceI(s,m1,m2), PhaseSpaceI(s,m3,m4);
    q02, q12 = kallen(s,m1**2,m2**2)/s/4., kallen(s,m3**2,m4**2)/s/4.;
    q0, q1   = np.sqrt(q02), np.sqrt(q12);
    rho0, rho1 = q0/np.sqrt(s)/8./np.pi, q1/np.sqrt(s)/8./np.pi
    G0II, G1II = G0-2.*1j*rho0, G1-2.*1j*rho1
    if   irs==1: H0, H1 = G0,   G1;
    elif irs==2: H0, H1 = G0II, G1;
    elif irs==3: H0, H1 = G0II, G1II;
    elif irs==4: H0, H1 = G0,   G1II;
    else: sys.exit('Wrong Riemann sheet')
    return (1.+H0*K00)*(1.+H1*K11)-H0*H1*K012;

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

def Tamp(si,l,m1,m2,m3,m4,m5,m6,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l):
    G0, G1 = PhaseSpace(si,m3,m4), PhaseSpace(si,m5,m6);
    s = si + 1j*0.00000001
    p2 = kallen(s,m1**2,m2**2)/s/4.
    q02, q12 = kallen(s,m3**2,m4**2)/s/4., kallen(s,m5**2,m6**2)/s/4.;
    p, q0, q1 = np.sqrt(p2), np.sqrt(q02), np.sqrt(q12);
    K00  = (q02**l)*( a00l + b00l*q02 )
    K01  = ((q0*q1)**l)*( a01l + b01l*q0*q1 )
    K11  = (q12**l)*( a11l + b11l*q12 )
    K012 = ((q02*q12)**l)*( a01l*a01l + b01l*b01l*q02*q12 + 2.*a01l*b01l*q0*q1 )
    if l==0:
        K00  = ( a00l + b00l*q02 )
        K01  = ( a01l + b01l*q0*q1 )
        K11  = ( a11l + b11l*q12 )
        K012 = ( a01l*a01l + b01l*b01l*q02*q12 + 2.*a01l*b01l*q0*q1 )
    DeltaK = K00*K11-K012
    D = (1.+G0*K00)*(1.+G1*K11)-G0*G1*K012
    A00 = (K00+G1*DeltaK)/D
    return A00;

def Amp(si,l,m1,m2,m3,m4,m5,m6,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l):
    G0, G1 = PhaseSpace(si,m3,m4), PhaseSpace(si,m5,m6);
    s = si + 1j*0.00000001
    p2 = kallen(s,m1**2,m2**2)/s/4.
    q02, q12 = kallen(s,m3**2,m4**2)/s/4., kallen(s,m5**2,m6**2)/s/4.;
    p, q0, q1 = np.sqrt(p2), np.sqrt(q02), np.sqrt(q12);
    K00  = (q02**l)*( a00l + b00l*q02 )
    K01  = ((q0*q1)**l)*( a01l + b01l*q0*q1 )
    K11  = (q12**l)*( a11l + b11l*q12 )
    K012 = ((q02*q12)**l)*( a01l*a01l + b01l*b01l*q02*q12 + 2.*a01l*b01l*q0*q1 )
    if l==0:
        K00  = ( a00l + b00l*q02 )
        K01  = ( a01l + b01l*q0*q1 )
        K11  = ( a11l + b11l*q12 )
        K012 = ( a01l*a01l + b01l*b01l*q02*q12 + 2.*a01l*b01l*q0*q1 )

    DeltaK = K00*K11-K012
    D = (1.+G0*K00)*(1.+G1*K11)-G0*G1*K012
    N0 = n0l*(p*q0)**l
    N1 = n1l*(p*q1)**l
    A00, A01 = (K00+G1*DeltaK)/D, K01/D;
    return N0*(1.-G0*A00) - N1*G1*A01;

def BcalL(s,t,l,m1,m2,m3,m4,m5,m6,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l):
    x = costhetafromt(s,t,m1,m2,m3,m4)
    Lpol = LegPol(l,x)
    return (2.*l+1.)*Lpol*Amp(s,l,m1,m2,m3,m4,m5,m6,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l); 

def Bcal(s,t,m1,m2,m3,m4,m5,m6,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l,lmax):
    return np.sum([ BcalL(s,t,l,m1,m2,m3,m4,m5,m6,n0l[l],n1l[l],a00l[l],a01l[l],a11l[l],b00l[l],b01l[l],b11l[l]) for l in range(lmax+1)])

def singleBcal(s,t,m1,m2,m3,m4,m5,m6,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l,l):
    return BcalL(s,t,l,m1,m2,m3,m4,m5,m6,n0l[l],n1l[l],a00l[l],a01l[l],a11l[l],b00l[l],b01l[l],b11l[l])

###############################################################################
#   Observables
###############################################################################

def dsigmadt_cc(s,t,m1,m2,m3,m4,m5,m6,N,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l,lmax):
    amplitude = Bcal(s,t,m1,m2,m3,m4,m5,m6,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l,lmax)
    num = np.absolute(amplitude)**2
    den = 16.*np.pi*(s-m2**2)**2
    return hbarc2*num*N/den;

def single_dsigmadt_cc(s,t,m1,m2,m3,m4,m5,m6,N,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l,l):
    amplitude = singleBcal(s,t,m1,m2,m3,m4,m5,m6,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l,l)
    num = np.absolute(amplitude)**2
    den = 16.*np.pi*(s-m2**2)**2
    return hbarc2*num*N/den;

def sigma_cc(s,m1,m2,m3,m4,m5,m6,N,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l,lmax):
    p, q = momentum(s,m1,m2), momentum(s,m3,m4)
    num = np.sum([ (2*l+1)*np.absolute(Amp(s,l,m1,m2,m3,m4,m5,m6,n0l[l],n1l[l],a00l[l],a01l[l],a11l[l],b00l[l],b01l[l],b11l[l]))**2 for l in range(lmax+1) ])
    den = 16.*np.pi*p*s
    return hbarc2*num*q*N/den;

def single_sigma_cc(s,m1,m2,m3,m4,m5,m6,N,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l,l):
    p, q = momentum(s,m1,m2), momentum(s,m3,m4)
    num = (2*l+1)*np.absolute(Amp(s,l,m1,m2,m3,m4,m5,m6,n0l[l],n1l[l],a00l[l],a01l[l],a11l[l],b00l[l],b01l[l],b11l[l]))**2
#    num = (2*l+1)*np.imag(Amp(s,l,m1,m2,m3,m4,m5,m6,n0l[l],n1l[l],a00l[l],a01l[l],a11l[l],b00l[l],b01l[l],b11l[l]))
    den = 16.*np.pi*p*s
    return hbarc2*num*q*N/den;

def sigma_total(s,m1,m2,m3,m4,m5,m6,N,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l,lmax):
    den = np.sqrt(kallen(s,m3**2,m4**2))
    num = np.sum([(2*l+1)*np.imag(Tamp(s,l,m1,m2,m3,m4,m5,m6,n0l[l],n1l[l],a00l[l],a01l[l],a11l[l],b00l[l],b01l[l],b11l[l])) for l in range(lmax+1) ])
#    for l in range(lmax+1):
#        print(s,l,hbarc2*(2*l+1)*np.imag(Tamp(s,l,m1,m2,m3,m4,m5,m6,n0l[l],n1l[l],a00l[l],a01l[l],a11l[l],b00l[l],b01l[l],b11l[l]))/den/1.0e6)
#    print(hbarc2*num/den/1.0e6)
    return hbarc2*num/den/1.0e6;

def observable_cc(s,t,m1,m2,m3,m4,m5,m6,N,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l,lmax,clase):
    if   clase==0: return sigma_cc(s,m1,m2,m3,m4,m5,m6,N,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l,lmax);
    elif clase==1: return dsigmadt_cc(s,t,m1,m2,m3,m4,m5,m6,N,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l,lmax);
    else: sys.exit('Wrong class')
    return 0;

###############################################################################
#   BS observables
###############################################################################

def bs_sigma_cc(xbs,sarray,m1,m2,m3,m4,m5,m6):
    ns, nbs  = len(sarray), len(xbs[:,0])
    idown68, iup68 = int(np.trunc(0.16*nbs)),  int(np.trunc(0.84*nbs))
    idown95, iup95 = int(np.trunc(0.025*nbs)), int(np.trunc(0.975*nbs))    
    avg, dw68, up68, dw95, up95 = np.zeros(ns), np.zeros(ns), np.zeros(ns), np.zeros(ns), np.zeros(ns);
    for j in range(ns):
        print(j+1,'xs out of',ns)
        s = sarray[j]
        xsec = np.zeros(nbs)
        for ibs in range(nbs):
            input0 = xbs[ibs,:]
            N = input0[2]
            parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
            n0l, n1l , a00l, a01l, a11l, b00l, b01l, b11l = np.array_split(parameters_input,8)
            lmax = len(n0l)-1
            xsec[ibs] = sigma_cc(s,m1,m2,m3,m4,m5,m6,N,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l,lmax)
        xsecsorted = np.sort(xsec)
        avg[j] = np.mean(xsecsorted)
        dw68[j], up68[j], dw95[j], up95[j] = xsecsorted[idown68], xsecsorted[iup68], xsecsorted[idown95], xsecsorted[iup95]
    return avg, dw68, up68, dw95, up95

def bs_dsigmadt_cc(xbs,s,tarray,m1,m2,m3,m4,m5,m6):
    nt, nbs  = len(tarray), len(xbs[:,0])
    idown68, iup68 = int(np.trunc(0.16*nbs)),  int(np.trunc(0.84*nbs))
    idown95, iup95 = int(np.trunc(0.025*nbs)), int(np.trunc(0.975*nbs))    
    avg, dw68, up68, dw95, up95 = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
    for k in range(nt):
        print(k+1,'dsdt out of',nt)
        t = tarray[k]
        dsdt = np.zeros(nbs)
        for ibs in range(nbs):
            input0 = xbs[ibs,:]
            N = input0[2]
            parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
            n0l, n1l, a00l, a01l, a11l, b00l, b01l, b11l = np.array_split(parameters_input,8)
            lmax = len(n0l)-1
            dsdt[ibs] = dsigmadt_cc(s,t,m1,m2,m3,m4,m5,m6,N,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l,lmax)
        dsdtsorted = np.sort(dsdt)
        avg[k] = np.mean(dsdtsorted)
        dw68[k], up68[k], dw95[k], up95[k] = dsdtsorted[idown68], dsdtsorted[iup68], dsdtsorted[idown95], dsdtsorted[iup95]
    return avg, dw68, up68, dw95, up95

def bs_total(xbs,sarray,m1,m2,m3,m4,m5,m6):
    ns, nbs  = len(sarray), len(xbs[:,0])
    idown68, iup68 = int(np.trunc(0.16*nbs)),  int(np.trunc(0.84*nbs))
    idown95, iup95 = int(np.trunc(0.025*nbs)), int(np.trunc(0.975*nbs))    
#    idown95, iup95 = int(np.trunc(0.05*nbs)), int(np.trunc(0.95*nbs))    
    dw68, up68, dw95, up95 = np.zeros(ns), np.zeros(ns), np.zeros(ns), np.zeros(ns);
    for j in range(ns):
        s = sarray[j]
        xsec = np.zeros(nbs)
        for ibs in range(nbs):
            input0 = xbs[ibs,:]
            N = input0[2]
            parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
            n0l, n1l , a00l, a01l, a11l, b00l, b01l, b11l = np.array_split(parameters_input,8)
            lmax = len(n0l)-1
            xsec[ibs] = sigma_total(s,m1,m2,m3,m4,m5,m6,N,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l,lmax)
        xsecsorted = np.sort(xsec)
        dw68[j], up68[j], dw95[j], up95[j] = xsecsorted[idown68], xsecsorted[iup68], xsecsorted[idown95], xsecsorted[iup95]
    return (dw68+up68)/2., dw68, up68, dw95, up95

###############################################################################
#   Fitting routine for MINUIT
###############################################################################

def LSQ_cc(par):
    m1, m2, m3, m4, m5, m6 = mphoton, mproton, mpsi, mproton, mdbar, mlambdac;
    N, parreduced = par[0], np.delete(par,0)
    n0l, n1l, a00l, a01l, a11l, b00l, b01l, b11l = np.array_split(parreduced,8)
    lmax = len(n0l) - 1
    s, t = sfromEbeam(Data.ebeam,m2), Data.t
    clase = Data.clase
    func = [ observable_cc(s[i],t[i],m1,m2,m3,m4,m5,m6,N,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l,lmax,clase[i]) for i in range(len(Data.ebeam))]
    return np.sum(((Data.obs-func)**2)/(Data.error**2))

def pull_cc(par):
    m1, m2, m3, m4, m5, m6 = mphoton, mproton, mpsi, mproton, mdbar, mlambdac;
    N, parreduced = par[0], np.delete(par,0)
    n0l, n1l, a00l, a01l, a11l, b00l, b01l, b11l = np.array_split(parreduced,8)
    lmax = len(n0l) - 1
    s, t = sfromEbeam(Data.ebeam,m2), Data.t
    clase = Data.clase
    func = [ observable_cc(s[i],t[i],m1,m2,m3,m4,m5,m6,N,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l,lmax,clase[i]) for i in range(len(Data.ebeam))]
    return (Data.obs-func)/Data.error

###############################################################################
#   Bootstrap dataset
###############################################################################

def pseudodataset(ydata,y_error):
    pseudodata = [ np.random.normal(ydata[i],y_error[i]) for i in np.arange(y_error.size)]
    return pseudodata

###############################################################################
#   Randomize input
###############################################################################

def input_generator(linput,lmax,ipar,rango,fixated,tmp_tmp):
    lini, lfin, ld = linput+1, lmax+1, 1
    tmp = tmp_tmp
    for ll in range(len(tmp_tmp)): ipar=ipar+1
    for ll in range(lini,lfin):
        ipar = ipar + 1
#        print(ipar,fixated[ipar])
        if fixated[ipar]==0:
            tmp1 = np.random.uniform(-rango,rango,ld)
        else:
            if ipar==1:
                tmp1 = np.ones(ld)
            else:
                tmp1 = np.zeros(ld)
        tmp  = np.concatenate((tmp_tmp,tmp1),axis=0)
        tmp_tmp = tmp
    return ipar, tmp

def input_generatorG(linput,lmax,ipar,rango,fixated,tmp_tmp):
    lini, lfin, ld = linput+1, lmax+1, 1
    tmp = copy.copy(tmp_tmp)
    for ll in range(len(tmp_tmp)):
        ipar=ipar+1
        if fixated[ipar]==0:
            if ipar==1: # Hack
                tmp1 = np.random.normal(0.,rango,ld)
            else:
                tmp1 = np.random.normal(tmp_tmp[ll],np.abs(tmp_tmp[ll]),ld)
            tmp[ll] = tmp1
    for ll in range(lini,lfin):
        ipar = ipar + 1
        if fixated[ipar]==0:
            tmp1 = np.random.normal(0.,rango,ld)
        else:
            tmp1 = np.zeros(ld)
        tmp = np.concatenate((tmp,tmp1),axis=0)
    return ipar, tmp

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

Datainput_gluexXsec = Namespace(clase=class_sigmagluex, eavg=Eavg_sigmagluex,
                            ebeam=Ebeam_sigmagluex, emin=Emin_sigmagluex, emax=Emax_sigmagluex,
                            t=t_sigmagluex, tmin=tmin_sigmagluex, tmax=tmax_sigmagluex,
                            obs=sigma_sigmagluex, error=error_sigmagluex)
npoints_gluexXsec = len(Datainput_gluexXsec.clase)


#   GlueX cross section cut: Class 0
gr = range(8)
#gr = range(12)
#gr = range(npoints_gluexXsec)
Datainput_gluexXsec_cut = Namespace(clase=np.take(class_sigmagluex,gr), 
                            eavg=np.take(Eavg_sigmagluex,gr), ebeam=np.take(Ebeam_sigmagluex,gr), 
                            emin=np.take(Emin_sigmagluex,gr), emax=np.take(Emax_sigmagluex,gr),
                            t=np.take(t_sigmagluex,gr), tmin=np.take(tmin_sigmagluex,gr), 
                            tmax=np.take(tmax_sigmagluex,gr),
                            obs=np.take(sigma_sigmagluex,gr), 
                            error=np.take(error_sigmagluex,gr))
npoints_gluexXsec_cut = len(Datainput_gluexXsec_cut.clase)


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
if   dataset=='gluexXsec':
    Datainput = Datainput_gluexXsec_cut;
    Normalization = normalization_gluex
elif dataset == 'gluex':
    Datainput = Datainput_gluex;
    Normalization = normalization_gluex 
elif dataset == '007':      
    Datainput = Datainput_007;
    Normalization = normalization_007 
elif dataset == 'combined': 
    Datainput = Datainput_comb;
    Normalization = normalization_comb    
else: sys.exit('Wrong dataset')

Data = Datainput
nclass = [np.count_nonzero(Datainput.clase==0), np.count_nonzero(Datainput.clase==1)]
ndata = np.sum(nclass)

if option in opciones and modelo in modelos:
    print('Calculation:',option)
    print('Dataset:',dataset)
    print('Number of datapoints:',ndata)
    print('Number of datapoints per class:',nclass)
else:
    sys.exit('Wrong option')    

#   Naming of parameters
vacio = ['N']
for i in range(lmax+1): vacio.append('n0'+str(i))
for i in range(lmax+1): vacio.append('n1'+str(i))
for i in range(lmax+1): vacio.append('a00'+str(i))
for i in range(lmax+1): vacio.append('a01'+str(i))
for i in range(lmax+1): vacio.append('a11'+str(i))
for i in range(lmax+1): vacio.append('b00'+str(i))
for i in range(lmax+1): vacio.append('b01'+str(i))
for i in range(lmax+1): vacio.append('b11'+str(i))
nombre = tuple( vacio[i] for i in range(len(vacio)) )

###############################################################################
#   Fitting. Exploring parameter space
###############################################################################

if option=='read' and ninputs==7:
    bff    = np.loadtxt(bffinput)
    input0 = bff[nmc,:]
    Ninput_tmp = input0[2]
    parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
    n0linput_tmp, n1linput_tmp, a00linput_tmp, a01linput_tmp, a11linput_tmp, b00linput_tmp, b01linput_tmp, b11linput_tmp = np.array_split(parameters_input,8)
    linput = len(n0linput_tmp)-1
    lmax = linput
    print('Lmax=',lmax)

    vacio = ['N']
    for i in range(lmax+1): vacio.append('n0'+str(i))
    for i in range(lmax+1): vacio.append('n1'+str(i))
    for i in range(lmax+1): vacio.append('a00'+str(i))
    for i in range(lmax+1): vacio.append('a01'+str(i))
    for i in range(lmax+1): vacio.append('a11'+str(i))
    for i in range(lmax+1): vacio.append('b00'+str(i))
    for i in range(lmax+1): vacio.append('b01'+str(i))
    for i in range(lmax+1): vacio.append('b11'+str(i))
    nombre = tuple( vacio[i] for i in range(len(vacio)) )

    for i in range(len(vacio)):
        print(vacio[i]+'=',input0[i+2])

    parameters_input = np.array([ input0[i] for i in range(2,len(input0))])
    chi2 = LSQ_cc(parameters_input)
    print('chi2=',chi2,'; chi2/N=',chi2/ndata)
    pull = pull_cc(parameters_input)
    print('Pull:')
    print(pull)
    print('Average pull=',np.mean(pull),'; Standard deviation=',np.std(pull))

elif option=='fit':

    #   Range for the random seed of the parameters
    rango = 1.

    #   Naming and fixing
    if modelo in ['init']:
        nome = 'N'
        vacio = ['N']; print(nome); inp = input()
        fixated = [int(inp)]
        for i in range(lmax+1):
            nome = 'n0'+str(i)
            vacio.append(nome); print(nome); inp = int(input())
            fixated.append(inp)
        for i in range(lmax+1):
            nome = 'n1'+str(i)
            vacio.append(nome); print(nome); inp = int(input())
            fixated.append(inp)
        for i in range(lmax+1): 
            nome = 'a00'+str(i)
            vacio.append(nome); print(nome); inp = int(input())
            fixated.append(inp)
        for i in range(lmax+1): 
            nome = 'a01'+str(i)
            vacio.append(nome); print(nome); inp = int(input())
            fixated.append(inp)
        for i in range(lmax+1): 
            nome = 'a11'+str(i)
            vacio.append(nome); print(nome); inp = int(input())
            fixated.append(inp)
        for i in range(lmax+1): 
            nome = 'b00'+str(i)
            vacio.append(nome); print(nome); inp = int(input())
            fixated.append(inp)
        for i in range(lmax+1): 
            nome = 'b01'+str(i)
            vacio.append(nome); print(nome); inp = int(input())
            fixated.append(inp)
        for i in range(lmax+1): 
            nome = 'b11'+str(i)
            vacio.append(nome); print(nome); inp = int(input())
            fixated.append(inp)
    else:
        vacio = ['N']
        for i in range(lmax+1): vacio.append('n0'+str(i))
        for i in range(lmax+1): vacio.append('n1'+str(i))
        for i in range(lmax+1): vacio.append('a00'+str(i))
        for i in range(lmax+1): vacio.append('a01'+str(i))
        for i in range(lmax+1): vacio.append('a11'+str(i))
        for i in range(lmax+1): vacio.append('b00'+str(i))
        for i in range(lmax+1): vacio.append('b01'+str(i))
        for i in range(lmax+1): vacio.append('b11'+str(i))
        nombre = tuple( vacio[i] for i in range(len(vacio)) )
        
        if modelo=='a':
            # N
            fixated = [0]
            # n0
            inp = 1
            fixated.append(inp)
            inp = 0
            for i in range(1,lmax+1): fixated.append(inp)
            # n1
            inp = 1
            for i in range(lmax+1): fixated.append(inp)
            inp = 0
            # a00
            for i in range(lmax+1): fixated.append(inp)
            # a01
            for i in range(lmax+1): fixated.append(inp)
            inp = 1
            # a11
            for i in range(lmax+1): fixated.append(inp)
            # b00
            for i in range(lmax+1): fixated.append(inp)
            # b01
            for i in range(lmax+1): fixated.append(inp)
            # b11
            for i in range(lmax+1): fixated.append(inp)

        elif modelo=='c':
            # N
            fixated = [0]
            # n0
            inp = 1
            for i in range(lmax+1): fixated.append(inp)
            inp = 0
            # n1
            for i in range(lmax+1): fixated.append(inp)
            # a00
            for i in range(lmax+1): fixated.append(inp)
            # a01
            for i in range(lmax+1): fixated.append(inp)
            # a11
            for i in range(lmax+1): fixated.append(inp)
            # b00
            inp = 1
            for i in range(lmax+1): fixated.append(inp)
            # b01
            for i in range(lmax+1): fixated.append(inp)
            # b11
            for i in range(lmax+1): fixated.append(inp)

    print('Lmax:',lmax)
    #   Number of free parameters
    npar = len(fixated)-np.sum(np.array(fixated))
    print('Number of parameters:',npar)

    #   Initialization of model parameters
    nn0l, nn1l = lmax+1, lmax+1
    na00l, na01l, na11l = lmax+1, lmax+1, lmax+1
    nb00l, nb01l, nb11l = lmax+1, lmax+1, lmax+1
    Ninput, n0linput, n1linput = [], [], [];
    a00linput, a01linput, a11linput= [], [], [];
    b00linput, b01linput, b11linput= [], [], [];
    if ninputs==7:
        bff    = np.loadtxt(bffinput)
        input0 = bff[0,:]
        Ninput_tmp = input0[2]
        parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
        n0linput_tmp, n1linput_tmp, a00linput_tmp, a01linput_tmp, a11linput_tmp, b00linput_tmp, b01linput_tmp, b11linput_tmp = np.array_split(parameters_input,8)
        linput = len(n0linput_tmp)-1
        print('Initial parameters:',Ninput_tmp,parameters_input )
        for i in range(nmc):
            ipar = 0
            #   N
            Ninput.append([Ninput_tmp])
            #   n0
            tmp0 = n0linput_tmp
            if modelo=='sfree': tmp0 = np.concatenate((np.random.uniform(-rango,rango,1),np.array([ n0linput_tmp[i] for i in range(1,len(n0linput_tmp-1))])),axis=0)
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            n0linput.append(tmp)
            #   n1
            tmp0 = n1linput_tmp
            if modelo=='sfree': tmp0 = np.concatenate((np.random.uniform(-rango,rango,1),np.array([ n1linput_tmp[i] for i in range(1,len(n1linput_tmp-1))])),axis=0)
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            n1linput.append(tmp)
            #   a00
            tmp0 = a00linput_tmp
            if modelo=='sfree': tmp0 = np.concatenate((np.random.uniform(-rango,rango,1),np.array([ a00linput_tmp[i] for i in range(1,len(a00linput_tmp-1))])),axis=0)
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            a00linput.append(tmp)
            #   a01
            tmp0 = a01linput_tmp
            if modelo=='sfree': tmp0 = np.concatenate((np.random.uniform(-rango,rango,1),np.array([ a01linput_tmp[i] for i in range(1,len(a01linput_tmp-1))])),axis=0)
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            a01linput.append(tmp)
            #   a11
            tmp0 = a11linput_tmp
            if modelo=='sfree': tmp0 = np.concatenate((np.random.uniform(-rango,rango,1),np.array([ a11linput_tmp[i] for i in range(1,len(a11linput_tmp-1))])),axis=0)
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            a11linput.append(tmp)
            #   b00
            tmp0 = b00linput_tmp
            if modelo=='sfree': tmp0 = np.concatenate((np.random.uniform(-rango,rango,1),np.array([ b00linput_tmp[i] for i in range(1,len(b00linput_tmp-1))])),axis=0)
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            b00linput.append(tmp)
            #   b01
            tmp0 = b01linput_tmp
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            #   b11
            tmp0 = b11linput_tmp
            if modelo=='sfree': tmp0 = np.concatenate((np.random.uniform(-rango,rango,1),np.array([ b11linput_tmp[i] for i in range(1,len(b11linput_tmp-1))])),axis=0)
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            b11linput.append(tmp)
    else:
        for i in range(nmc):
            ipar , linput, tmp0 = 0, -1, []
            #   N
            if fixated[ipar]==0:
                Ninput.append(np.random.uniform(-rango,rango,1))
            else:
                Ninput.append([1.])
            #   n0
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            n0linput.append(tmp)
            #   n1
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            n1linput.append(tmp)
            #   a00
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            a00linput.append(tmp)
            #   a01
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            a01linput.append(tmp)
            #   a11
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            a11linput.append(tmp)
            #   b00
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            b00linput.append(tmp)
            #   b01
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            b01linput.append(tmp)
            #   b11
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            b11linput.append(tmp)
    
    #   Fitting using MINUIT
    storage = []
    for i in range(nmc):
#        if i%10==0: print(i/nmc*100,'%')
        Nmc    = np.array(Ninput[i])
        n0lmc  = np.array(n0linput[i])
        n1lmc  = np.array(n1linput[i])
        a00lmc = np.array(a00linput[i])
        a01lmc = np.array(a01linput[i])
        a11lmc = np.array(a11linput[i])
        b00lmc = np.array(b00linput[i])
        b01lmc = np.array(b01linput[i])
        b11lmc = np.array(b11linput[i])
        print('initial n0',n0lmc)
        #print('Starting bees',b00lmc,b01lmc,b11lmc)
        parameters_input = np.concatenate((Nmc,n0lmc,n1lmc,a00lmc,a01lmc,a11lmc,b00lmc,b01lmc,b11lmc),axis=0)
        m_pc = Minuit(LSQ_cc,parameters_input,name=nombre)
        m_pc.errordef = Minuit.LEAST_SQUARES
        for kfix in range(len(fixated)): 
            if fixated[kfix]==1: m_pc.fixed[kfix] = True
        
        m_pc.migrad();
        chi2 = m_pc.fval
        chi2dof = chi2/(len(Datainput.obs)-npar)
        print(i+1,'chi2=',chi2,'chi2/dof=',chi2dof)
#        print(dashes); print(dashes);
        print(m_pc.params); 
#        print(m_pc.covariance); print(m_pc.covariance.correlation())
        N, parreduced = m_pc.values[0], np.delete(m_pc.values,0)
        n0l, n1l, a00l, a01l, a11l, b00l, b01l, b11l = np.array_split(parreduced,8)
        storage.append( (chi2,chi2dof,N,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l) )
        #   Structure boostrap fit = i
        #       chi2 = storage[i][0]
        #       chi2dof = storage[i][1]
        #       N = storage[i][2]
        #       al_0 = storagel[i][3][0], al_1 = storagel[i][3][1], ...
        #       bl_0 = storagel[i][4][0], al_1 = storagel[i][4][1], ...
    
    #   Sorting
    sorted_storage = sorted(storage, key=lambda chi2: chi2[0])
    
    #   File storage
    x_storage = []
    for i in range(nmc):
        x0, x1 = sorted_storage[i][0], sorted_storage[i][1]
        x2, x3 = sorted_storage[i][2], sorted_storage[i][3][:]
        x4, x5 = sorted_storage[i][4][:], sorted_storage[i][5][:]
        x6, x7 = sorted_storage[i][6][:], sorted_storage[i][7][:]   
        x8, x9 = sorted_storage[i][8][:], sorted_storage[i][9][:]
        x10 = sorted_storage[i][10][:]
        y0, y1 = [x0,x1,x2], np.concatenate((x3,x4,x5,x6,x7,x8,x9,x10),axis=0)
        if i==0: ybest = y1
        x = np.concatenate((y0,y1),axis=0)
        x_storage.append(x)
    
    np.savetxt('pcbff.txt', x_storage)  

###############################################################################
#   Bootstrap
###############################################################################

elif option=='bs':
        
    #   Naming and fixing
    if modelo=='init':
        nome = 'N'
        vacio = ['N']; print(nome); inp = input()
        fixated = [int(inp)]
        for i in range(lmax+1): 
            nome = 'n0'+str(i)
            vacio.append(nome); print(nome); inp = int(input())
            fixated.append(inp)
        for i in range(lmax+1):
            nome = 'n1'+str(i)
            vacio.append(nome); print(nome); inp = int(input())
            fixated.append(inp)
        for i in range(lmax+1): 
            nome = 'a00'+str(i)
            vacio.append(nome); print(nome); inp = int(input())
            fixated.append(inp)
        for i in range(lmax+1): 
            nome = 'a01'+str(i)
            vacio.append(nome); print(nome); inp = int(input())
            fixated.append(inp)
        for i in range(lmax+1): 
            nome = 'a11'+str(i)
            vacio.append(nome); print(nome); inp = int(input())
            fixated.append(inp)
        for i in range(lmax+1): 
            nome = 'b00'+str(i)
            vacio.append(nome); print(nome); inp = int(input())
            fixated.append(inp)
        for i in range(lmax+1): 
            nome = 'b01'+str(i)
            vacio.append(nome); print(nome); inp = int(input())
            fixated.append(inp)
        for i in range(lmax+1): 
            nome = 'b11'+str(i)
            vacio.append(nome); print(nome); inp = int(input())
            fixated.append(inp)
    elif modelo=='scat2':
        fixated = [ 1, #    N 
                   0, 0, 0, 0, #   n00 n01 n02 n03
                   0, 1, 1, 1, #   n10 n11 n12 n13 
                   0, 0, 0, 0, #   a000 a001 a002 a003
                   0, 1, 1, 1, #   a010 a011 a012 a013
                   0, 1, 1, 1, #   a110 a111 a112 a113
                   0, 1, 1, 1, #   b000 b001 b002 b003
                   1, 1, 1, 1, #   b010 b011 b012 b013
                   0, 1, 1, 1, #   b110 b111 b112 b113           
                   ]
        nome = 'N'
        vacio = ['N'];
        for i in range(lmax+1): 
            nome = 'n0'+str(i)
            vacio.append(nome);
        for i in range(lmax+1):
            nome = 'n1'+str(i)
            vacio.append(nome);
        for i in range(lmax+1): 
            nome = 'a00'+str(i)
            vacio.append(nome);
        for i in range(lmax+1): 
            nome = 'a01'+str(i)
            vacio.append(nome); 
        for i in range(lmax+1): 
            nome = 'a11'+str(i)
            vacio.append(nome); 
        for i in range(lmax+1): 
            nome = 'b00'+str(i)
            vacio.append(nome); 
        for i in range(lmax+1): 
            nome = 'b01'+str(i)
            vacio.append(nome); 
        for i in range(lmax+1): 
            nome = 'b11'+str(i)
            vacio.append(nome);
    else:
            vacio = ['N']
            for i in range(lmax+1): vacio.append('n0'+str(i))
            for i in range(lmax+1): vacio.append('n1'+str(i))
            for i in range(lmax+1): vacio.append('a00'+str(i))
            for i in range(lmax+1): vacio.append('a01'+str(i))
            for i in range(lmax+1): vacio.append('a11'+str(i))
            for i in range(lmax+1): vacio.append('b00'+str(i))
            for i in range(lmax+1): vacio.append('b01'+str(i))
            for i in range(lmax+1): vacio.append('b11'+str(i))
            nombre = tuple( vacio[i] for i in range(len(vacio)) )

    #print('Lmax:',lmax)
    #   Number of free parameters
    npar = len(fixated)-np.sum(np.array(fixated))
    #print('Number of parameters:',npar)

    nbs = nmc
        
    #   Initial values for the parameters
    if ninputs==7:
        bff    = np.loadtxt(bffinput)
    else:
        bff = np.loadtxt('pcbff.txt')
    input0 = bff[0,:]
    parameters_input = np.array([ input0[i] for i in range(2,len(input0))])
        
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
    #print('bsdata done')

    #   BS fits
    storage_bs = []
    for i in range(nbs):
        #print(i+1,'out of',nbs)
        Data.obs = np.array(ypseudodata[i])
        m_bs = Minuit(LSQ_cc,parameters_input,name=nombre)
        m_bs.errordef = Minuit.LEAST_SQUARES
        for kfix in range(len(fixated)): 
            if fixated[kfix]==1: m_bs.fixed[kfix] = True
        m_bs.migrad();
        chi2, chi2dof = m_bs.fval, m_bs.fval/(len(Datainput.obs)-npar);
        #print('BS Fit ',i+1,' out of ',nbs, chi2, chi2dof)
        #print(m_bs.params); 
        N, parreduced = m_bs.values[0], np.delete(m_bs.values,0)
        n0l, n1l, a00l, a01l, a11l, b00l, b01l, b11l = np.array_split(parreduced,8)
        storage_bs.append( (chi2,chi2dof,N,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l) )

    #   Sorting bs fits
    sorted_storage_bs = sorted(storage_bs, key=lambda chi2: chi2[0])
    
    #   BS storage
    x_storage = []
    for i in range(nbs):        
        x0, x1 = sorted_storage_bs[i][0], sorted_storage_bs[i][1]
        x2, x3 = sorted_storage_bs[i][2], sorted_storage_bs[i][3][:]
        x4, x5 = sorted_storage_bs[i][4][:], sorted_storage_bs[i][5][:]
        x6, x7 = sorted_storage_bs[i][6][:], sorted_storage_bs[i][7][:]  
        x8, x9 = sorted_storage_bs[i][8][:], sorted_storage_bs[i][9][:]  
        x10 = sorted_storage_bs[i][10][:]
        y0, y1 = [x0,x1,x2], np.concatenate((x3,x4,x5,x6,x7,x8,x9,x10),axis=0)
        x = np.concatenate((y0,y1),axis=0)
        x_storage.append(x)

    np.savetxt('pcbs.txt', x_storage)  

    #   Mean and errors
    na00l,  nn0l = len(a00l), len(n0l)
    down68, up68 = int(np.trunc(0.16*nbs)),  int(np.trunc(0.84*nbs))
    down95, up95 = int(np.trunc(0.025*nbs)), int(np.trunc(0.975*nbs))    
    al_storage = []

    j = 0
    Nl = np.array([ sorted_storage_bs[k][2] for k in range(nbs) ] )
    Nlsorted = np.sort(Nl)
    Nl_array = [ j, np.mean(Nlsorted), Nlsorted[down68], Nlsorted[up68], Nlsorted[down95], Nlsorted[up95] ]
    al_storage.append(Nl_array); j = j+1
    for i in range(nn0l):
        n0l = np.array([ sorted_storage_bs[k][3][i] for k in range(nbs) ] )
        n0lsorted = np.sort(n0l)
        n0l_array = [ j, np.mean(n0lsorted), n0lsorted[down68], n0lsorted[up68], n0lsorted[down95], n0lsorted[up95] ]
        al_storage.append(n0l_array)   
        j=j+1

        n1l = np.array([ sorted_storage_bs[k][4][i] for k in range(nbs) ] )
        n1lsorted = np.sort(n1l)
        n1l_array = [ j, np.mean(n1lsorted), n1lsorted[down68], n1lsorted[up68], n1lsorted[down95], n1lsorted[up95] ]
        al_storage.append(n1l_array); j=j+1

    for i in range(na00l):
        a00l = np.array([ sorted_storage_bs[k][5][i] for k in range(nbs) ] )
        a00lsorted = np.sort(a00l)
        a00l_array = [ j, np.mean(a00lsorted), a00lsorted[down68], a00lsorted[up68], a00lsorted[down95], a00lsorted[up95] ]
        al_storage.append(a00l_array); j=j+1
        
        a01l = np.array([ sorted_storage_bs[k][6][i] for k in range(nbs) ] )
        a01lsorted = np.sort(a01l)
        a01l_array = [ j, np.mean(a01lsorted), a01lsorted[down68], a01lsorted[up68], a01lsorted[down95], a01lsorted[up95] ]
        al_storage.append(a01l_array); j=j+1

        a11l = np.array([ sorted_storage_bs[k][7][i] for k in range(nbs) ] )
        a11lsorted = np.sort(a11l)
        a11l_array = [ j, np.mean(a11lsorted), a11lsorted[down68], a11lsorted[up68], a11lsorted[down95], a11lsorted[up95] ]
        al_storage.append(a11l_array); j=j+1

        b00l = np.array([ sorted_storage_bs[k][8][i] for k in range(nbs) ] )
        b00lsorted = np.sort(b00l)
        b00l_array = [ j, np.mean(b00lsorted), b00lsorted[down68], b00lsorted[up68], b00lsorted[down95], b00lsorted[up95] ]
        al_storage.append(b00l_array); j=j+1
        
        b01l = np.array([ sorted_storage_bs[k][9][i] for k in range(nbs) ] )
        b01lsorted = np.sort(b01l)
        b01l_array = [ j, np.mean(b01lsorted), b01lsorted[down68], b01lsorted[up68], b01lsorted[down95], b01lsorted[up95] ]
        al_storage.append(b01l_array); j=j+1

        b11l = np.array([ sorted_storage_bs[k][10][i] for k in range(nbs) ] )
        b11lsorted = np.sort(b11l)
        b11l_array = [ j, np.mean(b11lsorted), b11lsorted[down68], b11lsorted[up68], b11lsorted[down95], b11lsorted[up95] ]
        al_storage.append(b11l_array); j=j+1
        
    np.savetxt('pcmean_n_errors.txt', al_storage,fmt='%i %e %e %e %e %e')
    
    #   Covariance and correlation matrices
    #xarray = np.transpose(np.array(x_storage))    
    #xcovdiag = np.var(xarray, axis=1, ddof=1)
    #xcov, xcorr = np.cov(xarray), np.corrcoef(xarray);
    #np.savetxt('pccov.txt', xcov)  
    #np.savetxt('pccorr.txt', xcorr)
    
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

    sth = (mproton + mpsi)**2
    send = sfromEbeam(12.,mproton)
#    send = sfromEbeam(9.5,mproton)
    sarray = np.linspace(sth,send,1000)
    Earray = Ebeamfroms(sarray,mproton)

    if dataset in ['gluexXsec','gluex','combined']:
        
        xplots, yplots = 2, 2; 
        fig, subfig = plt.subplots(xplots,yplots,figsize=(15,15))
        xerror = (Emax_sigmagluex-Emin_sigmagluex)/2.
        
        if option=='plotlog':
            subfig[0,0].set_yscale('log')
            subfig[0,1].set_yscale('log')
            subfig[1,0].set_yscale('log')
            subfig[1,1].set_yscale('log')
            subfig[0,0].set_ylim((1e-4,5e0))
            subfig[0,1].set_ylim((1e-4,2e0))
            subfig[1,0].set_ylim((1e-4,2e0))
            subfig[1,1].set_ylim((1e-4,2e0))

        subfig[0,0].set_xlim((8,12.))
        subfig[0,1].set_xlim((0.,10.))
        subfig[1,0].set_xlim((0.,10.))
        subfig[1,1].set_xlim((0.,10.))

        subfig[0,0].errorbar(Ebeam_sigmagluex, sigma_sigmagluex, xerr=xerror, yerr=error_sigmagluex, fmt="o", markersize=3,capsize=5., c=jpac_color[9], alpha=1,zorder=3)
        for ifit in range(nini,nfin):
            input0 = bff[ifit,:]
            N = input0[2]
            parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
            n0l, n1l, a00l, a01l, a11l, b00l, b01l, b11l = np.array_split(parameters_input,8)
            lmax = len(a00l)-1
            xsec = [ sigma_cc(sarray[i],mphoton,mproton,mpsi,mproton,mdbar,mlambdac,N,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l,lmax) for i in range(len(sarray))]
            subfig[0,0].plot(Earray,xsec,'-',lw=2,c=jpac_color[0],alpha=1,zorder=2,label=r'$L_{max}$='+str(lmax))

            for l in range(lmax+1):
                pw2  = [ single_sigma_cc(sarray[i],mphoton,mproton,mpsi,mproton,mdbar,mlambdac,N,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l,l) for i in range(len(sarray))]
                subfig[0,0].plot(Earray,pw2,'--',lw=1,c=jpac_color[l],alpha=1,zorder=2,label=r'$\ell$='+str(l))
            
            n0l0 = np.zeros(lmax+1)
            n1l0 = np.zeros(lmax+1)
            pwn0 = [ single_sigma_cc(sarray[i],mphoton,mproton,mpsi,mproton,mdbar,mlambdac,N,n0l,n1l0,a00l,a01l,a11l,b00l,b01l,b11l,0) for i in range(len(sarray))]
            pwn1 = [ single_sigma_cc(sarray[i],mphoton,mproton,mpsi,mproton,mdbar,mlambdac,N,n0l0,n1l,a00l,a01l,a11l,b00l,b01l,b11l,0) for i in range(len(sarray))]
            
            subfig[0,0].plot(Earray,pwn0,'-',lw=1,c=jpac_color[8],alpha=1,zorder=2,label=r'$\ell=0, n_1^0=0$' )
            subfig[0,0].plot(Earray,pwn1,'-',lw=1,c=jpac_color[9],alpha=1,zorder=2,label=r'$\ell=0, n_0^0=0$' )
            
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

                dsdt = [ dsigmadt_cc(savg,tarray[k],mphoton,mproton,mpsi,mproton,mdbar,mlambdac,N,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l,lmax) for k in range(len(tarray))]
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
                    dsdt_pw = [ single_dsigmadt_cc(savg,tarray[k],mphoton,mproton,mpsi,mproton,mdbar,mlambdac,N,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l,l) for k in range(len(tarray))]
                    if l==0:
                        dsdt_pwn0 = [ single_dsigmadt_cc(savg,tarray[k],mphoton,mproton,mpsi,mproton,mdbar,mlambdac,N,n0l,n1l0,a00l,a01l,a11l,b00l,b01l,b11l,l) for k in range(len(tarray))]
                        dsdt_pwn1 = [ single_dsigmadt_cc(savg,tarray[k],mphoton,mproton,mpsi,mproton,mdbar,mlambdac,N,n0l0,n1l,a00l,a01l,a11l,b00l,b01l,b11l,l) for k in range(len(tarray))]

                    if i==0: 
                        if l==0:
                            subfig[0,1].plot(-tarray,dsdt_pwn0,'-',lw=1,c=jpac_color[8],alpha=1,zorder=2,label=r'$\ell=0, n_1^0=0$')
                            subfig[0,1].plot(-tarray,dsdt_pwn1,'-',lw=1,c=jpac_color[9],alpha=1,zorder=2,label=r'$\ell=0, n_0^0=0$')                     
                        subfig[0,1].plot(-tarray,dsdt_pw,'--',lw=1,c=jpac_color[l],alpha=1,zorder=2,label=r'$\ell$='+str(l))
                    elif i==1:
                        if l==0:
                            subfig[1,0].plot(-tarray,dsdt_pwn0,'-',lw=1,c=jpac_color[8],alpha=1,zorder=2,label=r'$\ell=0, n_1^0=0$')
                            subfig[1,0].plot(-tarray,dsdt_pwn1,'-',lw=1,c=jpac_color[9],alpha=1,zorder=2,label=r'$\ell=0, n_0^0=0$')                     
                        subfig[1,0].plot(-tarray,dsdt_pw,'--',lw=1,c=jpac_color[l],alpha=1,zorder=2,label=r'$\ell$='+str(l))
                    elif i==2:
                        if l==0:
                            subfig[1,1].plot(-tarray,dsdt_pwn0,'-',lw=1,c=jpac_color[8],alpha=1,zorder=2,label=r'$\ell=0, n_1^0=0$')
                            subfig[1,1].plot(-tarray,dsdt_pwn1,'-',lw=1,c=jpac_color[9],alpha=1,zorder=2,label=r'$\ell=0, n_0^0=0$')                     
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
            subfig[0,0].legend(loc='upper right',ncol=1,frameon=True,fontsize=11)
            subfig[0,1].legend(loc='upper right',ncol=1,frameon=True,fontsize=11)
            subfig[1,0].legend(loc='upper right',ncol=1,frameon=True,fontsize=11)
            subfig[1,1].legend(loc='upper right',ncol=1,frameon=True,fontsize=11)

        #plt.show()
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
#                    xerror = np.absolute(Datainput_007.tmin[ide]-Datainput_007.tmax[ide])/2.
                    yerror = Datainput_007.error[ide]
                    ebeam_text = str(Datainput_007.ebeam[ide])
#                    subfig[i,j].text(x,y,ebeam_text,fontsize=10)
                    subfig[i,j].errorbar(x,y,yerr=yerror, fmt="o", markersize=3,capsize=5., c=jpac_color[9], alpha=1,zorder=3)

                ebeam = Datainput_007.eavg[ide]
                savg = sfromEbeam(ebeam, mproton)
                tdw = tfromcostheta(savg,1.,mphoton,mproton,mpsi,mproton)
                tup = tfromcostheta(savg,-1.,mphoton,mproton,mpsi,mproton)
                tarray = np.linspace(tup,tdw,100)
                for ifit in range(nini,nfin):
                    input0 = bff[ifit,:]
                    N = input0[2]
                    parameters_input = np.array([ input0[ir] for ir in range(3,len(input0))])
                    n0l, n1l, a00l, a01l, a11l, b00l, b01l, b11l = np.array_split(parameters_input,8)
                    lmax = len(a00l)-1
                    dsdt = [ dsigmadt_cc(savg,tarray[ki],mphoton,mproton,mpsi,mproton,mdbar,mlambdac,N,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l,lmax) for ki in range(len(tarray))]
                    subfig[i,j].plot(-tarray,dsdt,'-',lw=2,c=jpac_color[0],alpha=1,zorder=2,label=r'$L_{max}$='+str(lmax))
                    for l in range(lmax+1):
                        dsdt_pw = [ single_dsigmadt_cc(savg,tarray[ki],mphoton,mproton,mpsi,mproton,mdbar,mlambdac,N,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l,l) for ki in range(len(tarray))]
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
        #plt.show()
        fig.savefig('plot007.pdf', bbox_inches='tight')
        
elif option=='plotbs' or option=='plotlogbs':

    nplotpoints = 100
    fuente = 20; 

    if modelo=='scat2':
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
        if modelo=='scat2':
            Earray, sarray, tarray, xsec, xsec_dw68, xsec_up68, xsec_dw95, xsec_up95 = xsec_file[0,:], xsec_file[1,:], xsec_file[2,:], xsec_file[3,:], xsec_file[4,:], xsec_file[5,:], xsec_file[6,:], xsec_file[7,:]
        else:            
            xsec, xsec_dw68, xsec_up68, xsec_dw95, xsec_up95 = bs_sigma_cc(bsf,sarray,mphoton,mproton,mpsi,mproton,mdbar,mlambdac)        
            storage_plot[0,:], storage_plot[1,:], storage_plot[2,:] = Earray, sarray, np.zeros(nplotpoints)
            storage_plot[3,:] = xsec
            storage_plot[4,:], storage_plot[5,:] = xsec_dw68, xsec_up68
            storage_plot[6,:], storage_plot[7,:] = xsec_dw95, xsec_up95
            np.savetxt('plot_xsec_gluex.txt', storage_plot)
            print('xsec computed and stored')

        xsec = (xsec_dw68 + xsec_up68)/2.
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
            tdw = tfromcostheta(savg, 1.,mphoton,mproton,mpsi,mproton)
            tup = tfromcostheta(savg,-1.,mphoton,mproton,mpsi,mproton)
            
            if modelo=='scat2':
                Earray, sarray, tarray, dsdt, dsdt_dw68, dsdt_up68, dsdt_dw95, dsdt_up95 = dsdt_file[i][0,:], dsdt_file[i][1,:], dsdt_file[i][2,:], dsdt_file[i][3,:], dsdt_file[i][4,:], dsdt_file[i][5,:], dsdt_file[i][6,:], dsdt_file[i][7,:]
            else:                        
                tarray = np.linspace(tup,tdw,100)
                dsdt, dsdt_dw68, dsdt_up68, dsdt_dw95, dsdt_up95 = bs_dsigmadt_cc(bsf,savg,tarray,mphoton,mproton,mpsi,mproton,mdbar,mlambdac)
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

                if modelo!='scat2':
                    np.savetxt('plot_dsdt_gluex_0.txt', storage_plot0)
                    print('first dsdt computed and stored')

            elif i==1:
                subfig[1,0].errorbar(-x, y, xerr=xerror, yerr=yerror, fmt="o", markersize=3,capsize=5., c=jpac_color[10], alpha=1,zorder=1)
                subfig[1,0].plot(-tarray,dsdt,'-',lw=2,c=jpac_color[0],alpha=1,zorder=1)
                subfig[1,0].fill_between(-tarray, new_dw68, new_up68, facecolor=jpac_color[0], interpolate=True, alpha=0.6,zorder=1)
                subfig[1,0].fill_between(-tarray, new_dw68, new_dw95, facecolor=jpac_color[2], interpolate=True, alpha=0.3,zorder=1)
                subfig[1,0].fill_between(-tarray, new_up68, new_up95, facecolor=jpac_color[2], interpolate=True, alpha=0.3,zorder=1)
                if modelo!='scat2':
                    np.savetxt('plot_dsdt_gluex_1.txt', storage_plot0)
                    print('second dsdt computed and stored')

            elif i==2:
                subfig[1,1].errorbar(-x, y, xerr=xerror, yerr=yerror, fmt="o", markersize=3,capsize=5., c=jpac_color[10], alpha=1,zorder=1)
                subfig[1,1].plot(-tarray,dsdt,'-',lw=2,c=jpac_color[0],alpha=1,zorder=1)
                subfig[1,1].fill_between(-tarray, new_dw68, new_up68, facecolor=jpac_color[0], interpolate=True, alpha=0.6,zorder=2)
                subfig[1,1].fill_between(-tarray, new_dw68, new_dw95, facecolor=jpac_color[2], interpolate=True, alpha=0.3,zorder=3)
                subfig[1,1].fill_between(-tarray, new_up68, new_up95, facecolor=jpac_color[2], interpolate=True, alpha=0.3,zorder=3)
                if modelo!='scat2':
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
#                    xerror = np.absolute(Datainput_007.tmin[ide]-Datainput_007.tmax[ide])/2.
                    yerror = Datainput_007.error[ide]
                    ebeam_text = str(Datainput_007.ebeam[ide])
#                    subfig[i,j].text(x,y,ebeam_text,fontsize=10)
                    subfig[i,j].errorbar(x,y,yerr=yerror, fmt="o", markersize=3,capsize=5., c=jpac_color[10], alpha=1,zorder=1)

                ebeam = Datainput_007.eavg[ide]
                savg = sfromEbeam(ebeam, mproton)
                tdw = tfromcostheta(savg, 1.,mphoton,mproton,mpsi,mproton)
                tup = tfromcostheta(savg,-1.,mphoton,mproton,mpsi,mproton)
                if modelo=='scat2':
                    Earray, sarray, tarray, dsdt, dsdt_dw68, dsdt_up68, dsdt_dw95, dsdt_up95 = dsdt_007_all[k][0,:], dsdt_007_all[k][1,:], dsdt_007_all[k][2,:], dsdt_007_all[k][3,:], dsdt_007_all[k][4,:], dsdt_007_all[k][5,:], dsdt_007_all[k][6,:], dsdt_007_all[k][7,:]
                else:                
                    tarray = np.linspace(tup,tdw,100)
                    dsdt, dsdt_dw68, dsdt_up68, dsdt_dw95, dsdt_up95 = bs_dsigmadt_cc(bsf,savg,tarray,mphoton,mproton,mpsi,mproton,mdbar,mlambdac)
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
        #plt.show()
        fig.savefig('plotbs007.pdf', bbox_inches='tight')
     
elif option=='test':
    lmax = 0
    Egam = 12.
    theta = np.pi/2.
    x = np.cos(theta)
    s = sfromEbeam(Egam,mproton)
    t = tfromcostheta(s,x,mphoton,mproton,mpsi,mproton)
    q0 = cmomentum(s,mpsi,mproton)
    q1 = cmomentum(s,mdbar,mlambdac)
    print(s,t,q0,q1)
    G0, G1 = PhaseSpace(s,mpsi,mproton), PhaseSpace(s,mdbar,mlambdac);
    print(G0,G1)
    N = 1.
    n0l  = [-0.09981451,-0.01462837,-0.00303203,-0.00069168 ]
    n1l  = [-3.1814832, 0., 0., 0. ]
    a00l = [-4.2314169, -0.87250104,-0.0357544,-0.11582914]
    a01l = [0.098642893, 0., 0., 0.]
    a11l = [0.94256618, 0., 0., 0.]
    b00l = [ -3.590315, 0., 0., 0.]
    b01l = [0.,0.,0.,0.]
    b11l = [-2.9318495,0.,0.,0.]
    m1, m2, m3, m4, m5, m6 = mphoton, mproton, mpsi, mproton, mdbar, mlambdac;

    l = 0
    dsdt = Amp(s,l,m1,m2,m3,m4,m5,m6,n0l[l],n1l[l],a00l[l],a01l[l],a11l[l],b00l[l],b01l[l],b11l[l])
    print(dsdt)
    
#    dsdt = dsigmadt_cc(s,t,mphoton,mproton,mpsi,mproton,mdbar,mlambdac,N,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l,lmax)
#    print(dsdt)
    
elif option=='polebff' or option=='polecheck':
    
    nini, nfin = nmc, lmax
    sth = (mproton + mpsi)**2
    xd, xu = sth-0.5, sfromEbeam(14.,mproton)
    yd, yu = 0.001, 1.
    x, y = np.linspace(xd,xu,5), np.linspace(yd,yu,5)
    stepx, stepy = x[1]-x[0], y[1]-y[0] 
    if ninputs==7:
        bff = np.loadtxt(bffinput)
    else:
        bff = np.loadtxt('pcbff.txt')

    if option=='polebff':
        hojas = [1,2,3,4]
    else:
        hojas = [1]
        
    nfits = len(bff[:,0])
    for ifit in range(nini,nfin):
        if option=='polebff': print(dashes)
        input0 = bff[ifit,:]
        chi2 = input0[0]
        parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
        n0l, n1l, a00l, a01l, a11l, b00l, b01l, b11l = np.array_split(parameters_input,8)
        lmax = len(a00l)-1
        npoles = 0
        for l in range(lmax+1):
            if option=='polebff': print('L=',l)
            a00, a01, a11 = a00l[l], a01l[l], a11l[l]
            b00, b01, b11 = b00l[l], b01l[l], b11l[l]
            for irs in hojas:
                for i in range(len(x)):
                    xdw = x[i]
                    xup = xdw + stepx
                    for j in range(len(y)): 
                        ydw = y[j]
                        yup = ydw + stepy
                        xa, xb = xdw - 1j*ydw , xup - 1j*yup 
                        polo, conv = csearch(Denominator,xa,xb,irs,mproton,mpsi,mdbar,mlambdac,a00,a01,a11,b00,b01,b11,l)
                        if polo.real>sth and np.abs(2*np.imag(np.sqrt(polo)))<0.5:
                            npoles = npoles + 1
                            if option=='polebff':
                                if polo.imag<0 and irs in [1,2,3]:
                                    check = Denominator(polo,irs,mproton,mpsi,mdbar,mlambdac,a00,a01,a11,b00,b01,b11,l)
                                    print('RS=',irs,'Pole=',np.sqrt(polo),'M=',np.real(np.sqrt(polo)),'G=',-np.imag(2*np.sqrt(polo)),'Ack=',np.abs(check))
                                    print(Ebeamfroms(np.real(polo),mproton))
                                elif polo.imag>0 and irs in [4]:
                                    check = Denominator(polo,irs,mproton,mpsi,mdbar,mlambdac,a00,a01,a11,b00,b01,b11,l)
                                    print('RS=',irs,'Pole=',np.sqrt(polo),'M=',np.real(np.sqrt(polo)),'G=', np.imag(2*np.sqrt(polo)),'Ack=',np.abs(check))
        if npoles==0 and option=='polecheck': print(ifit,chi2)
        
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
    
#    nplotpoints = 2
#    sth  =  sfromEbeam(10.,mproton)
#    send = sfromEbeam(11.,mproton)

    sarray = np.linspace(sth,send,nplotpoints)
    Earray = Ebeamfroms(sarray,mproton)
    storage_plot = np.zeros((3,nplotpoints))
    for ifit in range(nini,nfin):
        input0 = bff[ifit,:]
        N = input0[2]
        parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
        n0l, n1l, a00l, a01l, a11l, b00l, b01l, b11l = np.array_split(parameters_input,8)
        lmax = len(a00l)-1        
        xsec = [ sigma_total(sarray[i],mphoton,mproton,mpsi,mproton,mdbar,mlambdac,N,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l,lmax) for i in range(len(sarray))]
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

    if modelo=='scat2':
        xsec_file  = np.loadtxt('plot_totalbs.txt')
    else:
        if ninputs==7:
            xbs = np.loadtxt(bffinput)
        else:
            xbs = np.loadtxt('pcbs.txt')
        nfits = len(xbs[:,0])
        print('Number of BS fits=',nfits)
        send = sfromEbeam(15.,mproton)
        sarray = np.linspace(sth,send,nplotpoints)
        Earray = Ebeamfroms(sarray,mproton)
        
    storage_plot = np.zeros((8,nplotpoints))

    if modelo=='scat2':
        Earray, sarray, tarray, xsec, xsec_dw68, xsec_up68, xsec_dw95, xsec_up95 = xsec_file[0,:], xsec_file[1,:], xsec_file[2,:], xsec_file[3,:], xsec_file[4,:], xsec_file[5,:], xsec_file[6,:], xsec_file[7,:]
    else:            
        xsec, xsec_dw68, xsec_up68, xsec_dw95, xsec_up95 = bs_total(xbs,sarray,mphoton,mproton,mpsi,mproton,mdbar,mlambdac)
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
    plt.yscale('log'); plt.ylim(10e-2, 10e3)
    #plt.ylim(0, 60);

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



