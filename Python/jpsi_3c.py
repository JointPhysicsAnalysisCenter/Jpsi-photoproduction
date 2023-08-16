#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 2023

Three channels scattering length

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

opciones = ['fit','bs','plot','plotlog','plotbs','plotlogbs','test','polebff','polecheck','read','meshgrid','polebs','total','totalbs']
modelos  = ['scat3','init','init_n','scat3_n']

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
def csearch(func,xa,xb,irs,m1,m2,m3,m4,m5,m6,a00,a11,a22,a01,a02,a12,l,tol=1e-12,max_iter=32):
    # Evaluating the function and its magnitude in the points xa and xb
    fa   = func(xa,irs,m1,m2,m3,m4,m5,m6,a00,a11,a22,a01,a02,a12,l)
    fa_m = np.abs(fa)
    fb   = func(xb,irs,m1,m2,m3,m4,m5,m6,a00,a11,a22,a01,a02,a12,l)
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
        f0   = func(x0,irs,m1,m2,m3,m4,m5,m6,a00,a11,a22,a01,a02,a12,l)
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

def Denominator(s,irs,m1,m2,m3,m4,m5,m6,a00,a11,a22,a01,a02,a12,l):
    q02, q12, q22 = kallen(s,m1**2,m2**2)/s/4., kallen(s,m3**2,m4**2)/s/4., kallen(s,m5**2,m6**2)/s/4.;
    q0, q1, q2 = np.sqrt(q02), np.sqrt(q12), np.sqrt(q22);
    G0, G1, G2 = PhaseSpaceI(s,m1,m2), PhaseSpaceI(s,m3,m4), PhaseSpaceI(s,m5,m6);
#    rho0, rho1, rho2 = 2.*q0/np.sqrt(s), 2.*q1/np.sqrt(s), 2.*q2/np.sqrt(s)

    rho0, rho1, rho2 = q0/np.sqrt(s)/8./np.pi, q1/np.sqrt(s)/8./np.pi, q2/np.sqrt(s)/8./np.pi

    G0II, G1II, G2II = G0-2.*1j*rho0, G1-2.*1j*rho1, G2-2.*1j*rho2
    K00, K11, K22 = (q02**l)*a00, (q12**l)*a11, (q22**l)*a22;
    K012, K022, K122 = ((q02*q12)**l)*a01*a01, ((q02*q22)**l)*a02*a02, ((q12*q22)**l)*a12*a12;
    K3 = ((q02*q12*q22)**l)*a12*a02*a01;
    if   irs==1: H0, H1, H2 = G0,   G1,   G2;    #000  <- (+++)
    elif irs==2: H0, H1, H2 = G0,   G1,   G2II;  #001 (++-)
    elif irs==3: H0, H1, H2 = G0,   G1II, G2;    #010 (+-+)
    elif irs==4: H0, H1, H2 = G0,   G1II, G2II;  #011 (+--)
    elif irs==5: H0, H1, H2 = G0II, G1,   G2;    #100  <- 2nd (-++)
    elif irs==6: H0, H1, H2 = G0II, G1,   G2II;  #101 (-+-)
    elif irs==7: H0, H1, H2 = G0II, G1II, G2;    #110  <- 3rd (--+)
    elif irs==8: H0, H1, H2 = G0II, G1II, G2II;  #111  <- 
    else: sys.exit('Wrong Riemann sheet')
    return (1.+H0*K00)*(1.+H1*K11)*(1.+H2*K22) - H0*H1*K012 - H0*H2*K022 - H1*H2*K122 + H0*H1*H2*( 2.*K3 - K11*K022 - K00*K122 - K22*K012 );

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

def Tamp(si,l,m00,m01,m1,m2,m3,m4,m5,m6,n0l,n1l,n2l,a00,a11,a22,a01,a02,a12):
    s  = si + 1j*0.00000001
    p2 = kallen(s,m00**2,m01**2)/s/4.
    q02, q12, q22 = kallen(s,m1**2,m2**2)/s/4., kallen(s,m3**2,m4**2)/s/4., kallen(s,m5**2,m6**2)/s/4.;
    p, q0, q1, q2 = np.sqrt(p2), np.sqrt(q02), np.sqrt(q12), np.sqrt(q22);
    G0, G1, G2 = PhaseSpace(s,m1,m2), PhaseSpace(s,m3,m4), PhaseSpace(s,m5,m6);
    K00, K11, K22 = (q02**l)*a00, (q12**l)*a11, (q22**l)*a22;
    K01, K02, K12 = ((q0*q1)**l)*a01, ((q0*q2)**l)*a02, ((q1*q2)**l)*a12;
    K012, K022, K122 = ((q02*q12)**l)*a01*a01, ((q02*q22)**l)*a02*a02, ((q12*q22)**l)*a12*a12;
    K3 = ((q02*q12*q22)**l)*a12*a02*a01
    A00 = K00 - G1*K012 - G2*K022 + G1*K00*K11 - G1*G2*K022*K11 + 2.*G1*G2*K01*K02*K12 - G1*G2*K00*K122 + G2*K00*K22 - G1*G2*K012*K22 + G1*G2*K00*K11*K22
    D = (1.+G0*K00)*(1.+G1*K11)*(1.+G2*K22) - G0*G1*K012 - G0*G2*K022 - G1*G2*K122 + G0*G1*G2*( 2.*K3 - K11*K022 - K00*K122 - K22*K012 );
    return A00/D;

def Amp(si,l,m00,m01,m1,m2,m3,m4,m5,m6,n0l,n1l,n2l,a00,a11,a22,a01,a02,a12):
    s  = si + 1j*0.00000001
    p2 = kallen(s,m00**2,m01**2)/s/4.
    q02, q12, q22 = kallen(s,m1**2,m2**2)/s/4., kallen(s,m3**2,m4**2)/s/4., kallen(s,m5**2,m6**2)/s/4.;
    p, q0, q1, q2 = np.sqrt(p2), np.sqrt(q02), np.sqrt(q12), np.sqrt(q22);
    G0, G1, G2 = PhaseSpace(s,m1,m2), PhaseSpace(s,m3,m4), PhaseSpace(s,m5,m6);
    K00, K11, K22 = (q02**l)*a00, (q12**l)*a11, (q22**l)*a22;
    K01, K02, K12 = ((q0*q1)**l)*a01, ((q0*q2)**l)*a02, ((q1*q2)**l)*a12;
    K012, K022, K122 = ((q02*q12)**l)*a01*a01, ((q02*q22)**l)*a02*a02, ((q12*q22)**l)*a12*a12;
    K3 = ((q02*q12*q22)**l)*a12*a02*a01
    N0, N1, N2 = n0l*(p*q0)**l, n1l*(p*q1)**l, n2l*(p*q2)**l
    A00, A01, A02 = (1.+G1*K11)*(1.+G2*K22)-G1*G2*K122, K01*(1.+G2*K22)-G2*K02*K12, K02*(1.+G1*K11)-G1*K01*K12; 
    D = (1.+G0*K00)*(1.+G1*K11)*(1.+G2*K22) - G0*G1*K012 - G0*G2*K022 - G1*G2*K122 + G0*G1*G2*( 2.*K3 - K11*K022 - K00*K122 - K22*K012 );
    return (N0*A00-N1*G1*A01-N2*G2*A02)/D;

def BcalL(s,t,l,m00,m01,m1,m2,m3,m4,m5,m6,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l):
    x = costhetafromt(s,t,m00,m01,m1,m2)
    Lpol = LegPol(l,x)
    return (2.*l+1.)*Lpol*Amp(s,l,m00,m01,m1,m2,m3,m4,m5,m6,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l);

def Bcal(s,t,m00,m01,m1,m2,m3,m4,m5,m6,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,lmax):
    return np.sum([ BcalL(s,t,l,m00,m01,m1,m2,m3,m4,m5,m6,n0l[l],n1l[l],n2l[l],a00l[l],a11l[l],a22l[l],a01l[l],a02l[l],a12l[l]) for l in range(lmax+1)])

def singleBcal(s,t,m00,m01,m1,m2,m3,m4,m5,m6,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,l):
    return BcalL(s,t,l,m00,m01,m1,m2,m3,m4,m5,m6,n0l[l],n1l[l],n2l[l],a00l[l],a11l[l],a22l[l],a01l[l],a02l[l],a12l[l])

def numerator(si,l,m00,m01,m1,m2,m3,m4,m5,m6,n0l,n1l,n2l,a00,a11,a22,a01,a02,a12):
    s  = si + 1j*0.00000001
    p2 = kallen(s,m00**2,m01**2)/s/4.
    q02, q12, q22 = kallen(s,m1**2,m2**2)/s/4., kallen(s,m3**2,m4**2)/s/4., kallen(s,m5**2,m6**2)/s/4.;
    p, q0, q1, q2 = np.sqrt(p2), np.sqrt(q02), np.sqrt(q12), np.sqrt(q22);
    G1, G2 = PhaseSpace(s,m3,m4), PhaseSpace(s,m5,m6);
    K11, K22 = (q12**l)*a11, (q22**l)*a22;
    K01, K02, K12 = ((q0*q1)**l)*a01, ((q0*q2)**l)*a02, ((q1*q2)**l)*a12;
    K122 = ((q12*q22)**l)*a12*a12;
    N0, N1, N2 = n0l*(p*q0)**l, n1l*(p*q1)**l, n2l*(p*q2)**l
    A00, A01, A02 = (1.+G1*K11)*(1.+G2*K22)-G1*G2*K122, K01*(1.+G2*K22)-G2*K02*K12, K02*(1.+G1*K11)-G1*K01*K12; 
    normalization = np.absolute(N0*A00-N1*G1*A01-N2*G2*A02) 
    term0, term1, term2 = np.absolute(N0*A00), np.absolute(N1*G1*A01), np.absolute(N2*G2*A02) 
    ratio0, ratio1, ratio2 = term0/normalization, term1/normalization, term2/normalization
    return normalization, term0, term1, term2, ratio0, ratio1, ratio2

###############################################################################
#   Observables
###############################################################################

def dsigmadt_cc(s,t,m00,m01,m1,m2,m3,m4,m5,m6,N,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,lmax):
    amplitude = Bcal(s,t,m00,m01,m1,m2,m3,m4,m5,m6,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,lmax)
    num = np.absolute(amplitude)**2
    den = 16.*np.pi*(s-m01**2)**2
    return hbarc2*num*N/den;

def single_dsigmadt_cc(s,t,m00,m01,m1,m2,m3,m4,m5,m6,N,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,l):
    amplitude = singleBcal(s,t,m00,m01,m1,m2,m3,m4,m5,m6,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,l)
    num = np.absolute(amplitude)**2
    den = 16.*np.pi*(s-m01**2)**2
    return hbarc2*num*N/den;

def sigma_cc(s,m00,m01,m1,m2,m3,m4,m5,m6,N,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,lmax):
    p, q = momentum(s,m00,m01), momentum(s,m1,m2)
    num = np.sum([ (2*l+1)*np.absolute(Amp(s,l,m00,m01,m1,m2,m3,m4,m5,m6,n0l[l],n1l[l],n2l[l],a00l[l],a11l[l],a22l[l],a01l[l],a02l[l],a12l[l]))**2 for l in range(lmax+1) ])
    den = 16.*np.pi*p*s
    return hbarc2*num*q*N/den;

def single_sigma_cc(s,m00,m01,m1,m2,m3,m4,m5,m6,N,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,l):
    p, q = momentum(s,m00,m01), momentum(s,m1,m2)
    num = (2*l+1)*np.absolute(Amp(s,l,m00,m01,m1,m2,m3,m4,m5,m6,n0l[l],n1l[l],n2l[l],a00l[l],a11l[l],a22l[l],a01l[l],a02l[l],a12l[l]))**2
    den = 16.*np.pi*p*s
    return hbarc2*num*q*N/den;

def sigma_total(s,m00,m01,m1,m2,m3,m4,m5,m6,N,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,lmax):
    den = np.sqrt(kallen(s,m1**2,m2**2))
    num = np.sum([(2*l+1)*np.imag(Tamp(s,l,m00,m01,m1,m2,m3,m4,m5,m6,n0l[l],n1l[l],n2l[l],a00l[l],a11l[l],a22l[l],a01l[l],a02l[l],a12l[l])) for l in range(lmax+1) ])
    return hbarc2*num/den/1.0e6;

def observable_cc(s,t,m00,m01,m1,m2,m3,m4,m5,m6,N,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,lmax,clase):
    if   clase==0: return sigma_cc(s,m00,m01,m1,m2,m3,m4,m5,m6,N,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,lmax);
    elif clase==1: return dsigmadt_cc(s,t,m00,m01,m1,m2,m3,m4,m5,m6,N,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,lmax);
    else: sys.exit('Wrong class')
    return 0;

###############################################################################
#   BS observables
###############################################################################

def bs_sigma_cc(xbs,sarray,m00,m01,m1,m2,m3,m4,m5,m6):
    ns, nbs  = len(sarray), len(xbs[:,0])
    idown68, iup68 = int(np.trunc(0.16*nbs)),  int(np.trunc(0.84*nbs))
    idown95, iup95 = int(np.trunc(0.025*nbs)), int(np.trunc(0.975*nbs))    
    avg, dw68, up68, dw95, up95 = np.zeros(ns), np.zeros(ns), np.zeros(ns), np.zeros(ns), np.zeros(ns);
    for j in range(ns):
        s = sarray[j]
        xsec = np.zeros(nbs)
        for ibs in range(nbs):
            input0 = xbs[ibs,:]
            N = input0[2]
            parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
            n0l, n1l, n2l, a00l, a11l, a22l, a01l, a02l, a12l = np.array_split(parameters_input,9)
            lmax = len(n0l)-1
            xsec[ibs] = sigma_cc(s,m00,m01,m1,m2,m3,m4,m5,m6,N,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,lmax)
        xsecsorted = np.sort(xsec)
        avg[j] = np.mean(xsecsorted)
        dw68[j], up68[j], dw95[j], up95[j] = xsecsorted[idown68], xsecsorted[iup68], xsecsorted[idown95], xsecsorted[iup95]
    return avg, dw68, up68, dw95, up95

def bs_dsigmadt_cc(xbs,s,tarray,m00,m01,m1,m2,m3,m4,m5,m6):
    nt, nbs  = len(tarray), len(xbs[:,0])
    idown68, iup68 = int(np.trunc(0.16*nbs)),  int(np.trunc(0.84*nbs))
    idown95, iup95 = int(np.trunc(0.025*nbs)), int(np.trunc(0.975*nbs))    
    avg, dw68, up68, dw95, up95 = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
    for k in range(nt):
        t = tarray[k]
        dsdt = np.zeros(nbs)
        for ibs in range(nbs):
            input0 = xbs[ibs,:]
            N = input0[2]
            parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
            n0l, n1l, n2l, a00l, a11l, a22l, a01l, a02l, a12l = np.array_split(parameters_input,9)
            lmax = len(n0l)-1
            dsdt[ibs] = dsigmadt_cc(s,t,m00,m01,m1,m2,m3,m4,m5,m6,N,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,lmax)
        dsdtsorted = np.sort(dsdt)
        avg[k] = np.mean(dsdtsorted)
        dw68[k], up68[k], dw95[k], up95[k] = dsdtsorted[idown68], dsdtsorted[iup68], dsdtsorted[idown95], dsdtsorted[iup95]
    return avg, dw68, up68, dw95, up95

def bs_single_sigma_cc(xbs,sarray,m00,m01,m1,m2,m3,m4,m5,m6,l,nchoice):
    ns, nbs  = len(sarray), len(xbs[:,0])
    idown68, iup68 = int(np.trunc(0.16*nbs)),  int(np.trunc(0.84*nbs))
    idown95, iup95 = int(np.trunc(0.025*nbs)), int(np.trunc(0.975*nbs))    
    avg, dw68, up68, dw95, up95 = np.zeros(ns), np.zeros(ns), np.zeros(ns), np.zeros(ns), np.zeros(ns);
    for j in range(ns):
        s = sarray[j]
        xsec = np.zeros(nbs)
        for ibs in range(nbs):
            input0 = xbs[ibs,:]
            N = input0[2]
            parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
            n0l, n1l, n2l, a00l, a11l, a22l, a01l, a02l, a12l = np.array_split(parameters_input,9)
            lmax = len(n0l)-1
            if nchoice==0:
                n1l, n2l = np.zeros(lmax+1), np.zeros(lmax+1)
            elif nchoice==1:
                n0l, n2l =  np.zeros(lmax+1), np.zeros(lmax+1)
            elif nchoice==2:
                n0l, n1l =  np.zeros(lmax+1), np.zeros(lmax+1)
            xsec[ibs] = single_sigma_cc(s,m00,m01,m1,m2,m3,m4,m5,m6,N,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,l)
        xsecsorted = np.sort(xsec)
        avg[j] = np.mean(xsecsorted)
        dw68[j], up68[j], dw95[j], up95[j] = xsecsorted[idown68], xsecsorted[iup68], xsecsorted[idown95], xsecsorted[iup95]
    return avg, dw68, up68, dw95, up95

def bs_single_dsigmadt_cc(xbs,s,tarray,m00,m01,m1,m2,m3,m4,m5,m6,l,nchoice):
    nt, nbs  = len(tarray), len(xbs[:,0])
    idown68, iup68 = int(np.trunc(0.16*nbs)),  int(np.trunc(0.84*nbs))
    idown95, iup95 = int(np.trunc(0.025*nbs)), int(np.trunc(0.975*nbs))    
    avg, dw68, up68, dw95, up95 = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
    for k in range(nt):
        t = tarray[k]
        dsdt = np.zeros(nbs)
        for ibs in range(nbs):
            input0 = xbs[ibs,:]
            N = input0[2]
            parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
            n0l, n1l, n2l, a00l, a11l, a22l, a01l, a02l, a12l = np.array_split(parameters_input,9)
            lmax = len(n0l)-1
            if nchoice==0:
                n1l, n2l = np.zeros(lmax+1), np.zeros(lmax+1)
            elif nchoice==1:
                n0l, n2l = np.zeros(lmax+1), np.zeros(lmax+1)
            elif nchoice==2:
                n0l, n1l = np.zeros(lmax+1), np.zeros(lmax+1)
            dsdt[ibs] = single_dsigmadt_cc(s,t,m00,m01,m1,m2,m3,m4,m5,m6,N,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,l)
        dsdtsorted = np.sort(dsdt)
        avg[k] = np.mean(dsdtsorted)
        dw68[k], up68[k], dw95[k], up95[k] = dsdtsorted[idown68], dsdtsorted[iup68], dsdtsorted[idown95], dsdtsorted[iup95]
    return avg, dw68, up68, dw95, up95

def bs_numerator (xbs,sarray,m00,m01,m1,m2,m3,m4,m5,m6):
    ns, nbs, l  = len(sarray), len(xbs[:,0]), 0
    idown68, iup68 = int(np.trunc(0.16*nbs)),  int(np.trunc(0.84*nbs))
    idown95, iup95 = int(np.trunc(0.025*nbs)), int(np.trunc(0.975*nbs))    
    Ndw68, Nup68, Ndw95, Nup95 = np.zeros(ns), np.zeros(ns), np.zeros(ns), np.zeros(ns)
    t0dw68, t0up68, t0dw95, t0up95 = np.zeros(ns), np.zeros(ns), np.zeros(ns), np.zeros(ns)
    t1dw68, t1up68, t1dw95, t1up95 = np.zeros(ns), np.zeros(ns), np.zeros(ns), np.zeros(ns)
    t2dw68, t2up68, t2dw95, t2up95 = np.zeros(ns), np.zeros(ns), np.zeros(ns), np.zeros(ns)
    r0dw68, r0up68, r0dw95, r0up95 = np.zeros(ns), np.zeros(ns), np.zeros(ns), np.zeros(ns)
    r1dw68, r1up68, r1dw95, r1up95 = np.zeros(ns), np.zeros(ns), np.zeros(ns), np.zeros(ns)
    r2dw68, r2up68, r2dw95, r2up95 = np.zeros(ns), np.zeros(ns), np.zeros(ns), np.zeros(ns)
    for j in range(ns):
        s = sarray[j]
        normalization, term0, term1, term2, ratio0, ratio1, ratio2 = np.zeros(nbs), np.zeros(nbs), np.zeros(nbs), np.zeros(nbs), np.zeros(nbs), np.zeros(nbs), np.zeros(nbs)
        for ibs in range(nbs):
            input0 = xbs[ibs,:]
            N = input0[2]
            parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
            n0l, n1l, n2l, a00l, a11l, a22l, a01l, a02l, a12l = np.array_split(parameters_input,9)
            n0l,n1l,n2l,a00,a11,a22,a01,a02,a12=n0l[0], n1l[0], n2l[0], a00l[0], a11l[0], a22l[0], a01l[0], a02l[0], a12l[0];
            normalization[ibs], term0[ibs], term1[ibs], term2[ibs], ratio0[ibs], ratio1[ibs], ratio2[ibs] = numerator(s,l,m00,m01,m1,m2,m3,m4,m5,m6,n0l,n1l,n2l,a00,a11,a22,a01,a02,a12)

        normalizationsorted, term0sorted, term1sorted, term2sorted, ratio0sorted, ratio1sorted, ratio2sorted = np.sort(normalization), np.sort(term0), np.sort(term1), np.sort(term2), np.sort(ratio0), np.sort(ratio1), np.sort(ratio2)
        Ndw68[j], Nup68[j], Ndw95[j], Nup95[j] = normalizationsorted[idown68], normalizationsorted[iup68], normalizationsorted[idown95], normalizationsorted
        t0dw68[j], t0up68[j], t0dw95[j], t0up95[j] = term0sorted[idown68], term0sorted[iup68], term0sorted[idown95], term0sorted[iup95]
        t1dw68[j], t1up68[j], t1dw95[j], t1up95[j] = term1sorted[idown68], term1sorted[iup68], term1sorted[idown95], term1sorted[iup95]
        t2dw68[j], t2up68[j], t2dw95[j], t2up95[j] = term2sorted[idown68], term2sorted[iup68], term2sorted[idown95], term2sorted[iup95] 
        r0dw68[j], r0up68[j], r0dw95[j], r0up95[j] = ratio0sorted[idown68], ratio0sorted[iup68], ratio0sorted[idown95], ratio0sorted[iup95] 
        r1dw68[j], r1up68[j], r1dw95[j], r1up95[j] = ratio1sorted[idown68], ratio1sorted[iup68], ratio1sorted[idown95], ratio1sorted[iup95]
        r2dw68[j], r2up68[j], r2dw95[j], r2up95[j] = ratio2sorted[idown68], ratio2sorted[iup68], ratio2sorted[idown95], ratio2sorted[iup95] 
    return Ndw68, Nup68, Ndw95, Nup95, t0dw68, t0up68, t0dw95, t0up95, t1dw68, t1up68, t1dw95, t1up95, t2dw68, t2up68, t2dw95, t2up95, r0dw68, r0up68, r0dw95, r0up95, r1dw68, r1up68, r1dw95, r1up95, r2dw68, r2up68, r2dw95, r2up95;

def bs_total(xbs,sarray,m00,m01,m1,m2,m3,m4,m5,m6):
    ns, nbs  = len(sarray), len(xbs[:,0])
    idown68, iup68 = int(np.trunc(0.16*nbs)),  int(np.trunc(0.84*nbs))
    idown95, iup95 = int(np.trunc(0.025*nbs)), int(np.trunc(0.975*nbs))        
    dw68, up68, dw95, up95 = np.zeros(ns), np.zeros(ns), np.zeros(ns), np.zeros(ns);
    for j in range(ns):
        s = sarray[j]
        xsec = np.zeros(nbs)
        for ibs in range(nbs):
            input0 = xbs[ibs,:]
            N = input0[2]
            parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
            n0l, n1l, n2l, a00l, a11l, a22l, a01l, a02l, a12l = np.array_split(parameters_input,9)
            lmax = len(n0l)-1
            xsec[ibs] = sigma_total(s,m00,m01,m1,m2,m3,m4,m5,m6,N,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,lmax)
        xsecsorted = np.sort(xsec)
        dw68[j], up68[j], dw95[j], up95[j] = xsecsorted[idown68], xsecsorted[iup68], xsecsorted[idown95], xsecsorted[iup95]
    return (dw68+up68)/2., dw68, up68, dw95, up95

###############################################################################
#   Fitting routine for MINUIT
###############################################################################

def LSQ_cc(par):
    m00, m01 = mphoton, mproton
    m1, m2, m3, m4, m5, m6 = mpsi, mproton, md, mlambdac, mdbar, mlambdac;
    N, parreduced = par[0], np.delete(par,0)
    n0l, n1l, n2l, a00l, a11l, a22l, a01l, a02l, a12l = np.array_split(parreduced,9)
    lmax = len(n0l) - 1
    s, t = sfromEbeam(Data.ebeam,m01), Data.t
    clase = Data.clase
    func = [ observable_cc(s[i],t[i],m00,m01,m1,m2,m3,m4,m5,m6,N,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,lmax,clase[i]) for i in range(len(Data.ebeam))]
    return np.sum(((Data.obs-func)**2)/(Data.error**2))

def pull_cc(par):
    m00, m01 = mphoton, mproton
    m1, m2, m3, m4, m5, m6 = mpsi, mproton, md, mlambdac, mdbar, mlambdac;
    N, parreduced = par[0], np.delete(par,0)
    n0l, n1l, n2l, a00l, a11l, a22l, a01l, a02l, a12l = np.array_split(parreduced,9)
    lmax = len(n0l) - 1
    s, t = sfromEbeam(Data.ebeam,m01), Data.t
    clase = Data.clase
    func = [ observable_cc(s[i],t[i],m00,m01,m1,m2,m3,m4,m5,m6,N,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,lmax,clase[i]) for i in range(len(Data.ebeam))]
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

def input_generatorU(linput,lmax,ipar,rango,fixated,tmp0):
    lini, lfin, ld = linput+1, lmax+1, 1
    tmp = tmp0
    for ll in range(len(tmp0)): ipar=ipar+1
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
        tmp  = np.concatenate((tmp0,tmp1),axis=0)
        tmp0 = tmp
    return ipar, tmp

def input_generatorG(linput,lmax,ipar,rango,fixated,tmp_tmp):
    lini, lfin, ld = linput+1, lmax+1, 1
    tmp = copy.copy(tmp_tmp)
    for ll in range(len(tmp_tmp)):
        ipar=ipar+1
        if fixated[ipar]==0:
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
dsdt_dsdt007, error_dsdt007 =  dsdt007[:,11], dsdt007[:,14]
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
for i in range(lmax+1): vacio.append('n2'+str(i))
for i in range(lmax+1): vacio.append('a00'+str(i))
for i in range(lmax+1): vacio.append('a11'+str(i))
for i in range(lmax+1): vacio.append('a22'+str(i))
for i in range(lmax+1): vacio.append('a01'+str(i))
for i in range(lmax+1): vacio.append('a02'+str(i))
for i in range(lmax+1): vacio.append('a12'+str(i))
nombre = tuple( vacio[i] for i in range(len(vacio)) )


#   Thresholds
thresholds_E = [Ebeamfroms((mproton+mpsi)**2,mproton),Ebeamfroms((md+mlambdac)**2,mproton),Ebeamfroms((mdbar+mlambdac)**2,mproton)]
polefound = [8.522336073637502 ,8.966201905550529,8.966204511164975]


###############################################################################
#   Read
###############################################################################

if option=='read' and ninputs==7:
    bff    = np.loadtxt(bffinput)
    input0 = bff[nmc,:]
    Ninput_tmp = input0[2]
    parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
    n0linput_tmp, n1linput_tmp, n2linput_tmp, a00linput_tmp, a11linput_tmp, a22linput_tmp, a01linput_tmp, a02linput_tmp, a12linput_tmp = np.array_split(parameters_input,9)
    linput = len(n0linput_tmp)-1
    lmax = linput
    print('Lmax=',lmax)

    print('Thresholds')
    print('W=',mproton+mpsi,md+mlambdac,mdbar+mlambdac)
    print('E_g=',thresholds_E)

    vacio = ['N']
    for i in range(lmax+1): vacio.append('n0'+str(i))
    for i in range(lmax+1): vacio.append('n1'+str(i))
    for i in range(lmax+1): vacio.append('n2'+str(i))
    for i in range(lmax+1): vacio.append('a00'+str(i))
    for i in range(lmax+1): vacio.append('a11'+str(i))
    for i in range(lmax+1): vacio.append('a22'+str(i))
    for i in range(lmax+1): vacio.append('a01'+str(i))
    for i in range(lmax+1): vacio.append('a02'+str(i))
    for i in range(lmax+1): vacio.append('a12'+str(i))
    nombre = tuple( vacio[i] for i in range(len(vacio)) )

    for i in range(len(vacio)):
        print(vacio[i]+'=',input0[i+2])

    parameters_input = np.array([ input0[i] for i in range(2,len(input0))])
    chi2 = LSQ_cc(parameters_input)
    print('chi2=',chi2,'chi2/N=',chi2/ndata)
    pull = pull_cc(parameters_input)
    print('Pull:',len(pull))
    print(pull)
    print('Average pull=',np.mean(pull),'; Standard deviation=',np.std(pull))
    print('Sum=',np.sum(pull**2))

###############################################################################
#   Fitting. Exploring parameter space
###############################################################################

elif option=='fit':
    #   Range for the random seed of the parameters
    rango = 1.

    #   Naming and fixing
    if modelo in ['scat3','init']:
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
            nome = 'n2'+str(i)
            vacio.append(nome); print(nome); inp = int(input())
            fixated.append(inp)
        for i in range(lmax+1): 
            nome = 'a00'+str(i)
            vacio.append(nome); print(nome); inp = int(input())
            fixated.append(inp)
        for i in range(lmax+1): 
            nome = 'a11'+str(i)
            vacio.append(nome); print(nome); inp = int(input())
            fixated.append(inp)
        for i in range(lmax+1): 
            nome = 'a22'+str(i)
            vacio.append(nome); print(nome); inp = int(input())
            fixated.append(inp)
        for i in range(lmax+1): 
            nome = 'a01'+str(i)
            vacio.append(nome); print(nome); inp = int(input())
            fixated.append(inp)
        for i in range(lmax+1): 
            nome = 'a02'+str(i)
            vacio.append(nome); print(nome); inp = int(input())
            fixated.append(inp)
        for i in range(lmax+1): 
            nome = 'a12'+str(i)
            vacio.append(nome); print(nome); inp = int(input())
            fixated.append(inp)

    else:
        vacio = ['N']
        for i in range(lmax+1): vacio.append('n0'+str(i))
        for i in range(lmax+1): vacio.append('n1'+str(i))
        for i in range(lmax+1): vacio.append('n2'+str(i))
        for i in range(lmax+1): vacio.append('a00'+str(i))
        for i in range(lmax+1): vacio.append('a11'+str(i))
        for i in range(lmax+1): vacio.append('a22'+str(i))
        for i in range(lmax+1): vacio.append('a01'+str(i))
        for i in range(lmax+1): vacio.append('a02'+str(i))
        for i in range(lmax+1): vacio.append('a12'+str(i))
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
            # n2
            inp = 1
            for i in range(lmax+1): fixated.append(inp)
            inp = 0
            # a00
            for i in range(lmax+1): fixated.append(inp)
            # a11
            for i in range(lmax+1): fixated.append(inp)
            inp = 1
            # a22
            for i in range(lmax+1): fixated.append(inp)
            # a01
            for i in range(lmax+1): fixated.append(inp)
            # a02
            for i in range(lmax+1): fixated.append(inp)
            # a12
            for i in range(lmax+1): fixated.append(inp)

    print('Lmax:',lmax)
    #   Number of free parameters
    npar = len(fixated)-np.sum(np.array(fixated))
    print('Number of parameters:',npar)

    #   Initialization of model parameters
    nn0l, nn1l, nn2l = lmax+1, lmax+1, lmax+1
    na00l, na11l, na22l = lmax+1, lmax+1, lmax+1
    na01l, na02l, na12l = lmax+1, lmax+1, lmax+1
    Ninput = [];
    n0linput,  n1linput,  n2linput  = [], [], [];
    a00linput, a11linput, a22linput = [], [], [];
    a01linput, a02linput, a12linput = [], [], [];
    if ninputs==7:
        bff    = np.loadtxt(bffinput)
        input0 = bff[0,:]
        Ninput_tmp = input0[2]
        parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
        n0linput_tmp, n1linput_tmp, n2linput_tmp, a00linput_tmp, a11linput_tmp, a22linput_tmp, a01linput_tmp, a02linput_tmp, a12linput_tmp = np.array_split(parameters_input,9)
        linput = len(n0linput_tmp)-1
        print('Initial parameters:',Ninput_tmp,parameters_input )
        for i in range(nmc):
            ipar = 0
            #   N
            Ninput.append([Ninput_tmp])
            #   n0
            tmp0 = n0linput_tmp
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            n0linput.append(tmp)
            #   n1
            tmp0 = n1linput_tmp
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            n1linput.append(tmp)
            #   n2
            tmp0 = n2linput_tmp
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            n2linput.append(tmp)
            #   a00
            tmp0 = a00linput_tmp
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            a00linput.append(tmp)
            #   a11
            tmp0 = a11linput_tmp
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            a11linput.append(tmp)
            #   a22
            tmp0 = a22linput_tmp
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            a22linput.append(tmp)
            #   a01
            tmp0 = a01linput_tmp
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            a01linput.append(tmp)
            #   a02
            tmp0 = a02linput_tmp
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            a02linput.append(tmp)
            #   a12
            tmp0 = a12linput_tmp
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            a12linput.append(tmp)
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
            #   n2
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            n2linput.append(tmp)
            #   a00
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            a00linput.append(tmp)
            #   a11
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            a11linput.append(tmp)
            #   a22
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            a22linput.append(tmp)
            #   a01
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            a01linput.append(tmp)
            #   a02
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            a02linput.append(tmp)
            #   a12
            ipar, tmp = input_generatorG(linput,lmax,ipar,rango,fixated,tmp0)
            a12linput.append(tmp)

    #   Fitting using MINUIT
    storage = []
    for i in range(nmc):
        Nmc    = np.array(Ninput[i])
        n0lmc  = np.array(n0linput[i])
        n1lmc  = np.array(n1linput[i])
        n2lmc  = np.array(n2linput[i])
        a00lmc = np.array(a00linput[i])
        a11lmc = np.array(a11linput[i])
        a22lmc = np.array(a22linput[i])
        a01lmc = np.array(a01linput[i])
        a02lmc = np.array(a02linput[i])
        a12lmc = np.array(a12linput[i])
        
        parameters_input = np.concatenate((Nmc,n0lmc,n1lmc,n2lmc,a00lmc,a11lmc,a22lmc,a01lmc,a02lmc,a12lmc),axis=0)
        m_pc = Minuit(LSQ_cc,parameters_input,name=nombre)
        m_pc.errordef = Minuit.LEAST_SQUARES
        for kfix in range(len(fixated)): 
            if fixated[kfix]==1: m_pc.fixed[kfix] = True
                
        m_pc.migrad();
        chi2 = m_pc.fval
        chi2dof = chi2/(len(Datainput.obs)-npar)
        print(i+1,'chi2=',chi2,'chi2/dof=',chi2dof)
        print(m_pc.params); 
        N, parreduced = m_pc.values[0], np.delete(m_pc.values,0)
        n0l, n1l, n2l, a00l, a11l, a22l, a01l, a02l, a12l = np.array_split(parreduced,9)
        storage.append( (chi2,chi2dof,N,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l) )
    
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
        x10, x11 = sorted_storage[i][10][:], sorted_storage[i][11][:]
        y0, y1 = [x0,x1,x2], np.concatenate((x3,x4,x5,x6,x7,x8,x9,x10,x11),axis=0)
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
            nome = 'n2'+str(i)
            vacio.append(nome); print(nome); inp = int(input())
            fixated.append(inp)
        for i in range(lmax+1): 
            nome = 'a00'+str(i)
            vacio.append(nome); print(nome); inp = int(input())
            fixated.append(inp)
        for i in range(lmax+1): 
            nome = 'a11'+str(i)
            vacio.append(nome); print(nome); inp = int(input())
            fixated.append(inp)
        for i in range(lmax+1): 
            nome = 'a22'+str(i)
            vacio.append(nome); print(nome); inp = int(input())
            fixated.append(inp)
        for i in range(lmax+1): 
            nome = 'a01'+str(i)
            vacio.append(nome); print(nome); inp = int(input())
            fixated.append(inp)
        for i in range(lmax+1): 
            nome = 'a02'+str(i)
            vacio.append(nome); print(nome); inp = int(input())
            fixated.append(inp)
        for i in range(lmax+1): 
            nome = 'a12'+str(i)
            vacio.append(nome); print(nome); inp = int(input())
            fixated.append(inp)
    elif modelo=='scat3':
        fixated = [ 1, #    N 
                   0, 0, 0, 0, #   n00 n01 n02 n03
                   0, 1, 1, 1, #   n10 n11 n12 n13 
                   0, 1, 1, 1, #   n20 n21 n22 n23 
                   0, 0, 0, 0, #   a000 a001 a002 a003
                   0, 1, 1, 1, #   a110 a111 a112 a113
                   0, 1, 1, 1, #   a220 a221 a222 a223
                   0, 1, 1, 1, #   a010 a011 a012 a013
                   0, 1, 1, 1, #   a020 a021 a022 a023
                   0, 1, 1, 1  #   a120 a121 a122 a123          
                   ]
    else:
        vacio = ['N']
        for i in range(lmax+1): vacio.append('n0'+str(i))
        for i in range(lmax+1): vacio.append('n1'+str(i))
        for i in range(lmax+1): vacio.append('n2'+str(i))
        for i in range(lmax+1): vacio.append('a00'+str(i))
        for i in range(lmax+1): vacio.append('a11'+str(i))
        for i in range(lmax+1): vacio.append('a22'+str(i))
        for i in range(lmax+1): vacio.append('a01'+str(i))
        for i in range(lmax+1): vacio.append('a02'+str(i))
        for i in range(lmax+1): vacio.append('a12'+str(i))
        nombre = tuple( vacio[i] for i in range(len(vacio)) )

    #print('Lmax:',lmax)
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
        Data.obs = np.array(ypseudodata[i])
        m_bs = Minuit(LSQ_cc,parameters_input,name=nombre)
        m_bs.errordef = Minuit.LEAST_SQUARES
        for kfix in range(len(fixated)): 
            if fixated[kfix]==1: m_bs.fixed[kfix] = True
        m_bs.migrad();
        chi2, chi2dof = m_bs.fval, m_bs.fval/(len(Datainput.obs)-npar);
        N, parreduced = m_bs.values[0], np.delete(m_bs.values,0)
        n0l, n1l, n2l, a00l, a11l, a22l, a01l, a02l, a12l = np.array_split(parreduced,9)
        storage_bs.append( (chi2,chi2dof,N,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l) )

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
        x10, x11 = sorted_storage_bs[i][10][:], sorted_storage_bs[i][11][:]
        y0, y1 = [x0,x1,x2], np.concatenate((x3,x4,x5,x6,x7,x8,x9,x10,x11),axis=0)
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

    sth = (mproton + mpsi)**2
    send = sfromEbeam(12.,mproton)
    sarray = np.linspace(sth,send,1000)
    Earray = Ebeamfroms(sarray,mproton)

    if dataset in ['gluexXsec','gluex','combined']:
        
        xplots, yplots = 2, 2; 
        fig, subfig = plt.subplots(xplots,yplots,figsize=(15,15))

        for ifit in range(nini,nfin):
            input0 = bff[ifit,:]
            N = input0[2]
            parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
            n0l, n1l, n2l, a00l, a11l, a22l, a01l, a02l, a12l = np.array_split(parameters_input,9)
            lmax = len(a00l)-1
            l = 0
            amplitudeS = np.array([ Amp(sarray[i],l,mphoton,mproton,mpsi,mproton,md,mlambdac,mdbar,mlambdac,n0l[l],n1l[l],n2l[l],a00l[l],a11l[l],a22l[l],a01l[l],a02l[l],a12l[l]) for i in range(len(sarray)) ])
            denS = np.array([ Denominator(sarray[i]+ 1j*0.00000001,1,mpsi,mproton,md,mlambdac,mdbar,mlambdac,a00l[l],a11l[l],a22l[l],a01l[l],a02l[l],a12l[l],l)  for i in range(len(sarray))])

            subfig[0,0].plot(Earray,np.real(amplitudeS),'-',lw=2,c=jpac_color[0],alpha=1,zorder=2,label=r'Re Amp, $L_{max}$='+str(lmax))
            subfig[0,1].plot(Earray,np.imag(amplitudeS),'-',lw=2,c=jpac_color[0],alpha=1,zorder=2,label=r'Im Amp, $L_{max}$='+str(lmax))
            subfig[1,0].plot(Earray,np.real(denS),'-',lw=2,c=jpac_color[0],alpha=1,zorder=2,label=r'Re Den, $L_{max}$='+str(lmax))
            subfig[1,1].plot(Earray,np.imag(denS),'-',lw=2,c=jpac_color[0],alpha=1,zorder=2,label=r'Im Den $L_{max}$='+str(lmax))

            subfig[0,0].set_xlabel(r'$E_\gamma$ (GeV)',fontsize=fuente)
            subfig[0,1].set_xlabel(r'$E_\gamma$ (GeV)',fontsize=fuente)
            subfig[1,0].set_xlabel(r'$E_\gamma$ (GeV)',fontsize=fuente)
            subfig[1,1].set_xlabel(r'$E_\gamma$ (GeV)',fontsize=fuente)
    
            subfig[0,0].set_xlim((8,12))
            subfig[0,1].set_xlim((8,12))
            subfig[1,0].set_xlim((8,12))
            subfig[1,1].set_xlim((8,12))
 
            subfig[0,0].legend(loc='lower right',ncol=1,frameon=True,fontsize=11)
            subfig[0,1].legend(loc='lower right',ncol=1,frameon=True,fontsize=11)
            subfig[1,0].legend(loc='lower right',ncol=1,frameon=True,fontsize=11)
            subfig[1,1].legend(loc='lower right',ncol=1,frameon=True,fontsize=11)

            fig.savefig('amplitude.pdf', bbox_inches='tight')
            
            xplots, yplots = 2, 2; 
            fig, subfig = plt.subplots(xplots,yplots,figsize=(15,15))

            lens = len(sarray)
            normalization, term0, term1, term2, ratio0, ratio1, ratio2 = np.zeros(lens), np.zeros(lens), np.zeros(lens), np.zeros(lens), np.zeros(lens), np.zeros(lens), np.zeros(lens)
            for i in range(lens):
                normalization[i], term0[i], term1[i], term2[i], ratio0[i], ratio1[i], ratio2[i] = numerator(sarray[i],l,mphoton,mproton,mpsi,mproton,md,mlambdac,mdbar,mlambdac,n0l[l],n1l[l],n2l[l],a00l[l],a11l[l],a22l[l],a01l[l],a02l[l],a12l[l])

            subfig[0,0].plot(Earray,normalization,'-',lw=1,c=jpac_color[0],alpha=1,zorder=2,label=r'|Numerator|')
    
            subfig[1,0].plot(Earray,term0,'-',lw=1,c=jpac_color[1],alpha=1,zorder=2,label=r'$|J/\psi p |$')
            subfig[1,0].plot(Earray,term1,'-',lw=1,c=jpac_color[2],alpha=1,zorder=2,label=r'$| \bar{D}\Lambda_c |$')
            subfig[1,0].plot(Earray,term2,'-',lw=1,c=jpac_color[3],alpha=1,zorder=2,label=r'$| \bar{D}^{*}\Lambda_c |$')
    
            subfig[1,1].plot(Earray,ratio0,'-',lw=1,c=jpac_color[1],alpha=1,zorder=2,label=r'$|J/\psi p |$/|Numerator|')
            subfig[1,1].plot(Earray,ratio1,'-',lw=1,c=jpac_color[2],alpha=1,zorder=2,label=r'$|\bar{D}\Lambda_c |$/|Numerator|')
            subfig[1,1].plot(Earray,ratio2,'-',lw=1,c=jpac_color[3],alpha=1,zorder=2,label=r'$|\bar{D}^{*}\Lambda_c |$/|Numerator|')


        subfig[0,0].set_xlim((8,12))
        subfig[0,1].set_xlim((8,12))
        subfig[1,0].set_xlim((8,12))
        subfig[1,1].set_xlim((8,12))
        
        subfig[0,0].set_xlabel(r'$E_\gamma$ (GeV)',fontsize=fuente)
        subfig[1,0].set_xlabel(r'$E_\gamma$ (GeV)',fontsize=fuente)
        subfig[0,1].set_xlabel(r'$E_\gamma$ (GeV)',fontsize=fuente)
        subfig[1,1].set_xlabel(r'$E_\gamma$ (GeV)',fontsize=fuente)

        subfig[0,0].legend(loc='upper left',ncol=1,frameon=True,fontsize=11)
        subfig[0,1].legend(loc='upper left',ncol=1,frameon=True,fontsize=11)
        subfig[1,0].legend(loc='upper left',ncol=1,frameon=True,fontsize=11)
        subfig[1,1].legend(loc='upper right',ncol=1,frameon=True,fontsize=11)

        fig.savefig('numeratorS.pdf', bbox_inches='tight')
        

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

        subfig[0,0].vlines(thresholds_E[0],1e-3,1e1,linestyles ="dotted", colors =jpac_color[9], alpha=1,zorder=3)
        subfig[0,0].vlines(thresholds_E[1],1e-3,1e1,linestyles ="dotted", colors =jpac_color[8], alpha=1,zorder=3)
        subfig[0,0].vlines(thresholds_E[2],1e-3,1e1,linestyles ="dotted", colors =jpac_color[7], alpha=1,zorder=3)

        subfig[0,0].vlines(polefound[0] ,1e-3,1e1,linestyles ="dashed", colors =jpac_color[1], alpha=1,zorder=3)
        subfig[0,0].vlines(polefound[1] ,1e-3,1e1,linestyles ="dashed", colors =jpac_color[1], alpha=1,zorder=3)
        subfig[0,0].vlines(polefound[2] ,1e-3,1e1,linestyles ="dashed", colors =jpac_color[1], alpha=1,zorder=3)

        for ifit in range(nini,nfin):
            input0 = bff[ifit,:]
            N = input0[2]
            parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
            n0l, n1l, n2l, a00l, a11l, a22l, a01l, a02l, a12l = np.array_split(parameters_input,9)
            lmax = len(a00l)-1
            xsec = [ sigma_cc(sarray[i],mphoton,mproton,mpsi,mproton,md,mlambdac,mdbar,mlambdac,N,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,lmax) for i in range(len(sarray))]
            subfig[0,0].plot(Earray,xsec,'-',lw=2,c=jpac_color[0],alpha=1,zorder=2,label=r'$L_{max}$='+str(lmax))
            for l in range(lmax+1):
                pw2  = [ single_sigma_cc(sarray[i],mphoton,mproton,mpsi,mproton,md,mlambdac,mdbar,mlambdac,N,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,l) for i in range(len(sarray))]
                subfig[0,0].plot(Earray,pw2,'--',lw=1,c=jpac_color[l],alpha=1,zorder=2,label=r'$\ell$='+str(l))
            
            n0l0, n1l0, n2l0 = np.zeros(lmax+1), np.zeros(lmax+1), np.zeros(lmax+1);
            pwn0 = [ single_sigma_cc(sarray[i],mphoton,mproton,mpsi,mproton,md,mlambdac,mdbar,mlambdac,N,n0l,n1l0,n2l0,a00l,a11l,a22l,a01l,a02l,a12l,0) for i in range(len(sarray))]
            pwn1 = [ single_sigma_cc(sarray[i],mphoton,mproton,mpsi,mproton,md,mlambdac,mdbar,mlambdac,N,n0l0,n1l,n2l0,a00l,a11l,a22l,a01l,a02l,a12l,0) for i in range(len(sarray))]
            pwn2 = [ single_sigma_cc(sarray[i],mphoton,mproton,mpsi,mproton,md,mlambdac,mdbar,mlambdac,N,n0l0,n1l0,n2l,a00l,a11l,a22l,a01l,a02l,a12l,0) for i in range(len(sarray))]
            
            subfig[0,0].plot(Earray,pwn0,'-',lw=1,c=jpac_color[9],alpha=1,zorder=2,label=r'$\ell=0, n_0^0$')
            subfig[0,0].plot(Earray,pwn1,'-',lw=1,c=jpac_color[8],alpha=1,zorder=2,label=r'$\ell=0, n_1^0$')
            subfig[0,0].plot(Earray,pwn2,'-',lw=1,c=jpac_color[7],alpha=1,zorder=2,label=r'$\ell=0, n_2^0$')
            
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

                dsdt = [ dsigmadt_cc(savg,tarray[k],mphoton,mproton,mpsi,mproton,md,mlambdac,mdbar,mlambdac,N,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,lmax) for k in range(len(tarray))]
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
                    dsdt_pw = [ single_dsigmadt_cc(savg,tarray[k],mphoton,mproton,mpsi,mproton,md,mlambdac,mdbar,mlambdac,N,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,l) for k in range(len(tarray))]
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
                    N = input0[2]
                    parameters_input = np.array([ input0[ir] for ir in range(3,len(input0))])
                    n0l, n1l, n2l, a00l, a11l, a22l, a01l, a02l, a12l = np.array_split(parameters_input,9)
                    lmax = len(a00l)-1
                    dsdt = [ dsigmadt_cc(savg,tarray[ki],mphoton,mproton,mpsi,mproton,md,mlambdac,mdbar,mlambdac,N,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,lmax) for ki in range(len(tarray))]
                    subfig[i,j].plot(-tarray,dsdt,'-',lw=2,c=jpac_color[0],alpha=1,zorder=2,label=r'$L_{max}$='+str(lmax))
                    for l in range(lmax+1):
                        dsdt_pw = [ single_dsigmadt_cc(savg,tarray[ki],mphoton,mproton,mpsi,mproton,md,mlambdac,mdbar,mlambdac,N,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,l) for ki in range(len(tarray))]
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
                k = k+1
        fig.savefig('plot007.pdf', bbox_inches='tight')
        
elif option=='plotbs' or option=='plotlogbs':

    nplotpoints = 100
    fuente = 20; 

    if modelo=='scat3':
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
    elif modelo=='scat3_n':
        if dataset=='gluex' or dataset=='combined':
            xsec_file  = np.loadtxt('plot_xsec_gluex.txt')
            xsec_file_n0  = np.loadtxt('plot_xsec_n0_gluex.txt')
            xsec_file_n1  = np.loadtxt('plot_xsec_n1_gluex.txt')
            xsec_file_n2  = np.loadtxt('plot_xsec_n2_gluex.txt')
        else:
            sys.exit('Wrong option')
        
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
    
    if dataset=='gluex' and modelo in ['init_n','scat3_n']:
        xplots, yplots = 2, 2; 
        fig, subfig = plt.subplots(xplots,yplots,figsize=(15,15))
        xerror = (Emax_sigmagluex-Emin_sigmagluex)/2.
        
        if modelo=='scat3_n':
            Earray, sarray, tarray, xsec, xsec_dw68, xsec_up68, xsec_dw95, xsec_up95 = xsec_file[0,:], xsec_file[1,:], xsec_file[2,:], xsec_file[3,:], xsec_file[4,:], xsec_file[5,:], xsec_file[6,:], xsec_file[7,:]
            Earray_n0, sarray_n0, tarray_n0, xsec_n0, xsec_dw68_n0, xsec_up68_n0, xsec_dw95_n0, xsec_up95_n0 = xsec_file_n0[0,:], xsec_file_n0[1,:], xsec_file_n0[2,:], xsec_file_n0[3,:], xsec_file_n0[4,:], xsec_file_n0[5,:], xsec_file_n0[6,:], xsec_file_n0[7,:]
            Earray_n1, sarray_n1, tarray_n1, xsec_n1, xsec_dw68_n1, xsec_up68_n1, xsec_dw95_n1, xsec_up95_n1 = xsec_file_n1[0,:], xsec_file_n1[1,:], xsec_file_n1[2,:], xsec_file_n1[3,:], xsec_file_n1[4,:], xsec_file_n1[5,:], xsec_file_n1[6,:], xsec_file_n1[7,:]
            Earray_n2, sarray_n2, tarray_n2, xsec_n2, xsec_dw68_n2, xsec_up68_n2, xsec_dw95_n2, xsec_up95_n2 = xsec_file_n2[0,:], xsec_file_n2[1,:], xsec_file_n2[2,:], xsec_file_n2[3,:], xsec_file_n2[4,:], xsec_file_n2[5,:], xsec_file_n2[6,:], xsec_file_n2[7,:]
        else:
            xsec, xsec_dw68, xsec_up68, xsec_dw95, xsec_up95 = bs_sigma_cc(bsf,sarray,mphoton,mproton,mpsi,mproton,md,mlambdac,mdbar,mlambdac)
            storage_plot[0,:], storage_plot[1,:], storage_plot[2,:] = Earray, sarray, np.zeros(nplotpoints)
            xsec = (xsec_up68 + xsec_dw68)/2.
            storage_plot[3,:] = xsec
            storage_plot[4,:], storage_plot[5,:] = xsec_dw68, xsec_up68
            storage_plot[6,:], storage_plot[7,:] = xsec_dw95, xsec_up95
            np.savetxt('plot_xsec_gluex.txt', storage_plot)

            lfix = 0
            
            nchoice = 0
            xsec_n0, xsec_dw68_n0, xsec_up68_n0, xsec_dw95_n0, xsec_up95_n0 = bs_single_sigma_cc(bsf,sarray,mphoton,mproton,mpsi,mproton,md,mlambdac,mdbar,mlambdac,lfix,nchoice)
            storage_plot[0,:], storage_plot[1,:], storage_plot[2,:] = Earray, sarray, np.zeros(nplotpoints)
            xsec_n0 = (xsec_up68_n0 + xsec_dw68_n0)/2.
            storage_plot[3,:] = xsec_n0
            storage_plot[4,:], storage_plot[5,:] = xsec_dw68_n0, xsec_up68_n0
            storage_plot[6,:], storage_plot[7,:] = xsec_dw95_n0, xsec_up95_n0
            np.savetxt('plot_xsec_n0_gluex.txt', storage_plot)
            
            nchoice = 1
            xsec_n1, xsec_dw68_n1, xsec_up68_n1, xsec_dw95_n1, xsec_up95_n1 = bs_single_sigma_cc(bsf,sarray,mphoton,mproton,mpsi,mproton,md,mlambdac,mdbar,mlambdac,lfix,nchoice)
            storage_plot[0,:], storage_plot[1,:], storage_plot[2,:] = Earray, sarray, np.zeros(nplotpoints)
            xsec_n1 = (xsec_up68_n1 + xsec_dw68_n1)/2.
            storage_plot[3,:] = xsec_n1
            storage_plot[4,:], storage_plot[5,:] = xsec_dw68_n1, xsec_up68_n1
            storage_plot[6,:], storage_plot[7,:] = xsec_dw95_n1, xsec_up95_n1
            np.savetxt('plot_xsec_n1_gluex.txt', storage_plot)

            nchoice = 2
            xsec_n2, xsec_dw68_n2, xsec_up68_n2, xsec_dw95_n2, xsec_up95_n2 = bs_single_sigma_cc(bsf,sarray,mphoton,mproton,mpsi,mproton,md,mlambdac,mdbar,mlambdac,lfix,nchoice)
            storage_plot[0,:], storage_plot[1,:], storage_plot[2,:] = Earray, sarray, np.zeros(nplotpoints)
            xsec_n2 = (xsec_up68_n2 + xsec_dw68_n2)/2.
            storage_plot[3,:] = xsec_n2
            storage_plot[4,:], storage_plot[5,:] = xsec_dw68_n2, xsec_up68_n2
            storage_plot[6,:], storage_plot[7,:] = xsec_dw95_n2, xsec_up95_n2
            np.savetxt('plot_xsec_n2_gluex.txt', storage_plot)

        if option=='plotlogbs':
            subfig[0,0].set_yscale('log')
            subfig[0,1].set_yscale('log')
            subfig[1,0].set_yscale('log')
            subfig[1,1].set_yscale('log')
                        
        subfig[0,0].set_xlim((8,12))
        subfig[0,1].set_xlim((8,12))
        subfig[1,0].set_xlim((8,12))
        subfig[1,1].set_xlim((8,12))

        if lmax==1:
            xsec = (xsec_up68 + xsec_dw68 )/2.
            xsec95 = (xsec_up95 + xsec_dw95 )/2.
            Delta68 = np.sqrt(xsec*xsec*0.039601 + ((xsec_up68 - xsec_dw68)/2. )**2)
            Delta95 = np.sqrt(xsec95*xsec95*0.153664 + ((xsec_up95 - xsec_dw95)/2. )**2)
            new_up68, new_dw68 = xsec + Delta68, xsec - Delta68
            new_up95, new_dw95 = xsec95 + Delta95, xsec95 - Delta95
        else:
            xsec =  (xsec_up68 + xsec_dw68 )/2.
            new_up68, new_dw68 = xsec_up68, xsec_dw68
            new_up95, new_dw95 = xsec_up95, xsec_dw95
            
        subfig[0,0].errorbar(Ebeam_sigmagluex, sigma_sigmagluex, xerr=xerror, yerr=error_sigmagluex, fmt="o", markersize=3,capsize=5., c=jpac_color[10], alpha=1,zorder=1)
        subfig[0,0].plot(Earray,xsec,'-',lw=2,c=jpac_color[0],alpha=1,zorder=1)
        subfig[0,0].fill_between(Earray, new_dw68, new_up68, facecolor=jpac_color[0], interpolate=True, alpha=0.6,zorder=2)
        subfig[0,0].fill_between(Earray, new_dw68, new_dw95, facecolor=jpac_color[2], interpolate=True, alpha=0.3,zorder=3)
        subfig[0,0].fill_between(Earray, new_up68, new_up95, facecolor=jpac_color[2], interpolate=True, alpha=0.3,zorder=3)

        if lmax==1:
            xsec_n0 =  (xsec_up68_n0+ xsec_dw68_n0)/2.
            xsec95 = (xsec_up95_n0 + xsec_dw95_n0)/2.
            Delta68 = np.sqrt(xsec_n0*xsec_n0*0.039601 + ((xsec_up68_n0 - xsec_dw68_n0)/2. )**2)
            Delta95 = np.sqrt(xsec95*xsec95*0.153664 + ((xsec_up95_n0 - xsec_dw95_n0)/2. )**2)
            new_up68, new_dw68 = xsec_n0 + Delta68, xsec_n0 - Delta68
            new_up95, new_dw95 = xsec95 + Delta95,  xsec95 - Delta95
        else:
            xsec_n0 =  (xsec_up68_n0+ xsec_dw68_n0)/2.
            new_up68, new_dw68 = xsec_up68_n0, xsec_dw68_n0
            new_up95, new_dw95 = xsec_up95_n0, xsec_dw95_n0
    
        subfig[0,1].errorbar(Ebeam_sigmagluex, sigma_sigmagluex, xerr=xerror, yerr=error_sigmagluex, fmt="o", markersize=3,capsize=5., c=jpac_color[10], alpha=1,zorder=1)
        subfig[0,1].plot(Earray,xsec_n0,'-',lw=2,c=jpac_color[0],alpha=1,zorder=1)
        subfig[0,1].fill_between(Earray, new_dw68, new_up68, facecolor=jpac_color[0], interpolate=True, alpha=0.6,zorder=2)
        subfig[0,1].fill_between(Earray, new_dw68, new_dw95, facecolor=jpac_color[2], interpolate=True, alpha=0.3,zorder=3)
        subfig[0,1].fill_between(Earray, new_up68, new_up95, facecolor=jpac_color[2], interpolate=True, alpha=0.3,zorder=3)

        if lmax==1:
            xsec_n1 =  (xsec_up68_n1+ xsec_dw68_n1)/2.
            xsec95 = (xsec_up95_n1 + xsec_dw95_n1)/2.
            Delta68 = np.sqrt(xsec_n1*xsec_n1*0.039601 + ((xsec_up68_n1 - xsec_dw68_n1)/2. )**2)
            Delta95 = np.sqrt(xsec95*xsec95*0.153664 + ((xsec_up95_n1 - xsec_dw95_n1)/2. )**2)
            new_up68, new_dw68 = xsec_n1 + Delta68, xsec_n1 - Delta68
            new_up95, new_dw95 = xsec95 + Delta95,  xsec95 - Delta95
        else:
            xsec_n1 =  (xsec_up68_n1+ xsec_dw68_n1)/2.
            new_up68, new_dw68 = xsec_up68_n1, xsec_dw68_n1
            new_up95, new_dw95 = xsec_up95_n1, xsec_dw95_n1

        subfig[1,0].errorbar(Ebeam_sigmagluex, sigma_sigmagluex, xerr=xerror, yerr=error_sigmagluex, fmt="o", markersize=3,capsize=5., c=jpac_color[10], alpha=1,zorder=1)
        subfig[1,0].plot(Earray,xsec_n1,'-',lw=2,c=jpac_color[0],alpha=1,zorder=1)
        subfig[1,0].fill_between(Earray, new_dw68, new_up68, facecolor=jpac_color[0], interpolate=True, alpha=0.6,zorder=2)
        subfig[1,0].fill_between(Earray, new_dw68, new_dw95, facecolor=jpac_color[2], interpolate=True, alpha=0.3,zorder=3)
        subfig[1,0].fill_between(Earray, new_up68, new_up95, facecolor=jpac_color[2], interpolate=True, alpha=0.3,zorder=3)

        if lmax==1:
            xsec_n2 =  (xsec_up68_n2+ xsec_dw68_n2)/2.
            xsec95 = (xsec_up95_n2 + xsec_dw95_n2)/2.
            Delta68 = np.sqrt(xsec_n2*xsec_n2*0.039601 + ((xsec_up68_n2 - xsec_dw68_n2)/2. )**2)
            Delta95 = np.sqrt(xsec95*xsec95*0.153664 + ((xsec_up95_n2 - xsec_dw95_n2)/2. )**2)
            new_up68, new_dw68 = xsec_n2 + Delta68, xsec_n2 - Delta68
            new_up95, new_dw95 = xsec95 + Delta95,  xsec95 - Delta95
        else:
            xsec_n2=  (xsec_up68_n2+ xsec_dw68_n2)/2.
            new_up68, new_dw68 = xsec_up68_n2, xsec_dw68_n2
            new_up95, new_dw95 = xsec_up95_n2, xsec_dw95_n2

        subfig[1,1].errorbar(Ebeam_sigmagluex, sigma_sigmagluex, xerr=xerror, yerr=error_sigmagluex, fmt="o", markersize=3,capsize=5., c=jpac_color[10], alpha=1,zorder=1)
        subfig[1,1].plot(Earray,xsec_n2,'-',lw=2,c=jpac_color[0],alpha=1,zorder=1)
        subfig[1,1].fill_between(Earray, new_dw68, new_up68, facecolor=jpac_color[0], interpolate=True, alpha=0.6,zorder=2)
        subfig[1,1].fill_between(Earray, new_dw68, new_dw95, facecolor=jpac_color[2], interpolate=True, alpha=0.3,zorder=3)
        subfig[1,1].fill_between(Earray, new_up68, new_up95, facecolor=jpac_color[2], interpolate=True, alpha=0.3,zorder=3)

        fig.savefig('plotbsgluex_n.pdf', bbox_inches='tight')

    if dataset in ['gluex','combined'] and modelo in ['init','scat3']:
        
        xplots, yplots = 2, 2; 
        fig, subfig = plt.subplots(xplots,yplots,figsize=(15,15))
        xerror = (Emax_sigmagluex-Emin_sigmagluex)/2.
        
        if modelo=='scat3':
            Earray, sarray, tarray, xsec, xsec_dw68, xsec_up68, xsec_dw95, xsec_up95 = xsec_file[0,:], xsec_file[1,:], xsec_file[2,:], xsec_file[3,:], xsec_file[4,:], xsec_file[5,:], xsec_file[6,:], xsec_file[7,:]
            xsec =  (xsec_up68 + xsec_dw68 )/2.
        else:            
            xsec, xsec_dw68, xsec_up68, xsec_dw95, xsec_up95 = bs_sigma_cc(bsf,sarray,mphoton,mproton,mpsi,mproton,md,mlambdac,mdbar,mlambdac)
            storage_plot[0,:], storage_plot[1,:], storage_plot[2,:] = Earray, sarray, np.zeros(nplotpoints)
            xsec = (xsec_up68 + xsec_dw68 )/2.
            storage_plot[3,:] = xsec
            storage_plot[4,:], storage_plot[5,:] = xsec_dw68, xsec_up68
            storage_plot[6,:], storage_plot[7,:] = xsec_dw95, xsec_up95
            np.savetxt('plot_xsec_gluex.txt', storage_plot)

        if option=='plotlogbs':
            subfig[0,0].set_yscale('log')
            subfig[0,1].set_yscale('log')
            subfig[1,0].set_yscale('log')
            subfig[1,1].set_yscale('log')

        subfig[0,0].set_xlim((8,12))
        subfig[0,1].set_xlim((0.,10.))
        subfig[1,0].set_xlim((0.,10.))
        subfig[1,1].set_xlim((0.,10.))

        if lmax==1:
            xsec =  (xsec_up68 + xsec_dw68 )/2.
            xsec95 = (xsec_up95 + xsec_dw95 )/2.
            Delta68 = np.sqrt(xsec*xsec*0.039601 + ((xsec_up68 - xsec_dw68)/2. )**2)
            Delta95 = np.sqrt(xsec95*xsec95*0.153664 + ((xsec_up95 - xsec_dw95)/2. )**2)
            new_up68, new_dw68 = xsec + Delta68, xsec - Delta68
            new_up95, new_dw95 = xsec95 + Delta95, xsec95 - Delta95
        else:
            xsec = (xsec_up68 + xsec_dw68 )/2.
            new_up68, new_dw68 = xsec_up68, xsec_dw68
            new_up95, new_dw95 = xsec_up95, xsec_dw95

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
            
            if modelo=='scat3':
                Earray, sarray, tarray, dsdt, dsdt_dw68, dsdt_up68, dsdt_dw95, dsdt_up95 = dsdt_file[i][0,:], dsdt_file[i][1,:], dsdt_file[i][2,:], dsdt_file[i][3,:], dsdt_file[i][4,:], dsdt_file[i][5,:], dsdt_file[i][6,:], dsdt_file[i][7,:]
                dsdt =  (dsdt_up68 + dsdt_dw68 )/2.
            else:                        
                tarray = np.linspace(tup,tdw,100)
                dsdt, dsdt_dw68, dsdt_up68, dsdt_dw95, dsdt_up95 = bs_dsigmadt_cc(bsf,savg,tarray,mphoton,mproton,mpsi,mproton,md,mlambdac,mdbar,mlambdac)
                storage_plot0[0,:], storage_plot0[1,:], storage_plot0[2,:] = np.full(nplotpoints,ebeam), np.full(nplotpoints,savg), tarray
                dsdt =  (dsdt_up68 + dsdt_dw68 )/2.
                storage_plot0[3,:] = dsdt
                storage_plot0[4,:], storage_plot0[5,:] = dsdt_dw68, dsdt_up68
                storage_plot0[6,:], storage_plot0[7,:] = dsdt_dw95, dsdt_up95

            if lmax==1:
                dsdt =  (dsdt_up68 + dsdt_dw68 )/2.
                dsdt95 =  (dsdt_up95 + dsdt_dw95 )/2.
                Delta68 = np.sqrt(dsdt*dsdt*0.039601 + ((dsdt_up68 - dsdt_dw68)/2. )**2)
                Delta95 = np.sqrt(dsdt95*dsdt95*0.153664 + ((dsdt_up95 - dsdt_dw95)/2. )**2)
                new_up68 = dsdt + Delta68
                new_dw68 = dsdt - Delta68
                new_up95 = dsdt95 + Delta95
                new_dw95 = dsdt95 - Delta95
            else:
                dsdt =  (dsdt_up68 + dsdt_dw68 )/2.
                new_up68, new_dw68 = dsdt_up68, dsdt_dw68
                new_up95, new_dw95 = dsdt_up95, dsdt_dw95

            if i==0: 
                subfig[0,1].errorbar(-x, y, xerr=xerror, yerr=yerror, fmt="o", markersize=3,capsize=5., c=jpac_color[10], alpha=1,zorder=1)
                subfig[0,1].plot(-tarray,dsdt,'-',lw=2,c=jpac_color[0],alpha=1,zorder=1)
                subfig[0,1].fill_between(-tarray, new_dw68, new_up68, facecolor=jpac_color[0], interpolate=True, alpha=0.6,zorder=1)
                subfig[0,1].fill_between(-tarray, new_dw68, new_dw95, facecolor=jpac_color[2], interpolate=True, alpha=0.3,zorder=1)
                subfig[0,1].fill_between(-tarray, new_up68, new_up95, facecolor=jpac_color[2], interpolate=True, alpha=0.3,zorder=1)

                if modelo!='scat3':
                    np.savetxt('plot_dsdt_gluex_0.txt', storage_plot0)
                    print('first dsdt computed and stored')

            elif i==1:
                subfig[1,0].errorbar(-x, y, xerr=xerror, yerr=yerror, fmt="o", markersize=3,capsize=5., c=jpac_color[10], alpha=1,zorder=1)
                subfig[1,0].plot(-tarray,dsdt,'-',lw=2,c=jpac_color[0],alpha=1,zorder=1)
                subfig[1,0].fill_between(-tarray, new_dw68, new_up68, facecolor=jpac_color[0], interpolate=True, alpha=0.6,zorder=1)
                subfig[1,0].fill_between(-tarray, new_dw68, new_dw95, facecolor=jpac_color[2], interpolate=True, alpha=0.3,zorder=1)
                subfig[1,0].fill_between(-tarray, new_up68, new_up95, facecolor=jpac_color[2], interpolate=True, alpha=0.3,zorder=1)
                if modelo!='scat3':
                    np.savetxt('plot_dsdt_gluex_1.txt', storage_plot0)
                    print('second dsdt computed and stored')

            elif i==2:
                subfig[1,1].errorbar(-x, y, xerr=xerror, yerr=yerror, fmt="o", markersize=3,capsize=5., c=jpac_color[10], alpha=1,zorder=1)
                subfig[1,1].plot(-tarray,dsdt,'-',lw=2,c=jpac_color[0],alpha=1,zorder=1)
                subfig[1,1].fill_between(-tarray, new_dw68, new_up68, facecolor=jpac_color[0], interpolate=True, alpha=0.6,zorder=2)
                subfig[1,1].fill_between(-tarray, new_dw68, new_dw95, facecolor=jpac_color[2], interpolate=True, alpha=0.3,zorder=3)
                subfig[1,1].fill_between(-tarray, new_up68, new_up95, facecolor=jpac_color[2], interpolate=True, alpha=0.3,zorder=3)
                if modelo!='scat3':
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
                tdw = tfromcostheta(savg, 1.,mphoton,mproton,mpsi,mproton)
                tup = tfromcostheta(savg,-1.,mphoton,mproton,mpsi,mproton)
                if modelo=='scat3':
                    Earray, sarray, tarray, dsdt, dsdt_dw68, dsdt_up68, dsdt_dw95, dsdt_up95 = dsdt_007_all[k][0,:], dsdt_007_all[k][1,:], dsdt_007_all[k][2,:], dsdt_007_all[k][3,:], dsdt_007_all[k][4,:], dsdt_007_all[k][5,:], dsdt_007_all[k][6,:], dsdt_007_all[k][7,:]
                    dsdt =  (dsdt_up68 + dsdt_dw68 )/2.
                else:                
                    tarray = np.linspace(tup,tdw,100)
                    dsdt, dsdt_dw68, dsdt_up68, dsdt_dw95, dsdt_up95 = bs_dsigmadt_cc(bsf,savg,tarray,mphoton,mproton,mpsi,mproton,md,mlambdac,mdbar,mlambdac)
                    storage_plot[0,:], storage_plot[1,:], storage_plot[2,:] = np.full(nplotpoints,ebeam), np.full(nplotpoints,savg), tarray
                    storage_plot[3,:] = dsdt
                    storage_plot[4,:], storage_plot[5,:] = dsdt_dw68, dsdt_up68
                    storage_plot[6,:], storage_plot[7,:] = dsdt_dw95, dsdt_up95
                    filestoragename = 'plot_dsdt_007'+str(k)+'.txt'
                    np.savetxt(filestoragename, storage_plot)

                if lmax==1:
                    dsdt =  (dsdt_up68 + dsdt_dw68 )/2.
                    dsdt95 =  (dsdt_up95 + dsdt_dw95 )/2.    
                    Delta68 = np.sqrt(dsdt*dsdt*0.0016 + ((dsdt_up68 - dsdt_dw68)/2. )**2)
                    Delta95 = np.sqrt(dsdt95*dsdt95*0.0064 + ((dsdt_up95 - dsdt_dw95)/2. )**2)
                    new_up68,new_dw68 = dsdt + Delta68, dsdt - Delta68
                    new_up95, new_dw95 = dsdt95 + Delta95, dsdt95 - Delta95
                else:
                    dsdt =  (dsdt_up68 + dsdt_dw68 )/2.
                    new_up68, new_dw68 = dsdt_up68, dsdt_dw68
                    new_up95, new_dw95 = dsdt_up95, dsdt_dw95

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
     
elif option=='test':
    print('outdated')
    
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
        hojas = [1,2,3,4,5,6,7,8]
    else:
        hojas = [1]
        
    nfits = len(bff[:,0])
    for ifit in range(nini,nfin):
        if option=='polebff': print(dashes)
        input0 = bff[ifit,:]
        chi2 = input0[0]
        parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
        n0l, n1l, n2l, a00l, a11l, a22l, a01l, a02l, a12l = np.array_split(parameters_input,9)
        lmax = len(a00l)-1
        npoles = 0
        for l in range(lmax+1):
            if option=='polebff': print('L=',l)
            a00, a11, a22 = a00l[l], a11l[l], a22l[l]
            a01, a02, a12 = a01l[l], a02l[l], a12l[l]
            for irs in hojas:
                for i in range(len(x)):
                    xdw = x[i]
                    xup = xdw + stepx
                    for j in range(len(y)): 
                        ydw = y[j]
                        yup = ydw + stepy
                        xa, xb = xdw - 1j*ydw , xup - 1j*yup 
                        polo, conv = csearch(Denominator,xa,xb,irs,mproton,mpsi,md,mlambdac,mdbar,mlambdac,a00,a11,a22,a01,a02,a12,l)
                        if polo.real>sth and np.abs(2*np.imag(np.sqrt(polo)))<0.5:
                            npoles = npoles + 1
                            if option=='polebff':
                                check = Denominator(polo,irs,mproton,mpsi,md,mlambdac,mdbar,mlambdac,a00,a11,a22,a01,a02,a12,l)
                                print('RS=',irs,'Pole=',np.sqrt(polo),r'$E_\gamma=$',Ebeamfroms(np.real(polo),mproton),'$M=$',np.real(np.sqrt(polo)),'$\Gamma=$',-np.imag(2*np.sqrt(polo)),'Ack=',np.abs(check))
        if npoles==0 and option=='polecheck': print(ifit,chi2)

elif option=='polebs':

    sth = (mproton + mpsi)**2
    hojas = [1,2,3,4,5,6,7,8]
    
    if modelo=='init':
        xd, xu = sth-0.5, sfromEbeam(14.,mproton)
        yd, yu = 0.001, 1.
        xa, xb = xd - 1j*yd , xu - 1j*yu
        l = lmax
        if ninputs==7:
            bsf = np.loadtxt(bffinput)
        else:
            bsf = np.loadtxt('pcbs.txt')
    
        nfits = len(bsf[:,0])
        print('Number of BS fits=',nfits)
        
        storage_poles = np.zeros((nfits,len(hojas)),dtype=complex)
        count = np.zeros(len(hojas))
        for ifit in range(nfits):
            input0 = bsf[ifit,:]
            chi2 = input0[0]
            parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
            n0l, n1l, n2l, a00l, a11l, a22l, a01l, a02l, a12l = np.array_split(parameters_input,9)
            a00, a11, a22 = a00l[l], a11l[l], a22l[l]
            a01, a02, a12 = a01l[l], a02l[l], a12l[l]
            for irs in hojas:
                polo, conv = csearch(Denominator,xa,xb,irs,mproton,mpsi,md,mlambdac,mdbar,mlambdac,a00,a11,a22,a01,a02,a12,l)
                if polo.real>sth and np.abs(2*np.imag(np.sqrt(polo)))<0.5:
                    check = Denominator(polo,irs,mproton,mpsi,md,mlambdac,mdbar,mlambdac,a00,a11,a22,a01,a02,a12,l)
                    if check<1e-12:
                        storage_poles[ifit,irs-1] = np.sqrt(polo)
                        count[irs] = count[irs] + 1
                    else:
                        storage_poles[ifit,irs-1] = 0. + 1j*0.
                else:
                    storage_poles[ifit,irs-1] = 0. + 1j*0.
        for irs in hojas:
            print('RS=',irs,'Number of poles=',count[irs-1])

        np.savetxt('polebs.txt', storage_poles)  
    else:
        if ninputs==7:
            storage_poles = np.loadtxt(bffinput)
        else:
            storage_poles = np.loadtxt('polebs.txt')

    fuente = 20
    xplots, yplots = 3, 3
    fig, subfig = plt.subplots(xplots,yplots,figsize=(7*yplots,7*xplots))

    NameRS = ['I (+++)','VI (++-)','V (+-+)','VIII (+--)','II (-++)','VII (-+-)','III (--+)','IV (---)']
    irs = 0
    for i in range(xplots):
        for j in range(yplots):
            irs = irs +1
            if irs in hojas:
                mass  = np.real(storage_poles[:,irs-1])
                width = -np.imag(2*storage_poles[:,irs-1])
                
                subfig[i,j].vlines(mproton+mpsi,-0.05,0.25,linestyles ="dotted", colors =jpac_color[9], alpha=1,zorder=3)
                subfig[i,j].vlines(md+mlambdac,-0.05,0.25,linestyles ="dotted", colors =jpac_color[8], alpha=1,zorder=3)
                subfig[i,j].vlines(mdbar+mlambdac,-0.05,0.25,linestyles ="dotted", colors =jpac_color[7], alpha=1,zorder=3)

                subfig[i,j].scatter(mass,width,s=3,c=jpac_color[0],alpha=1,zorder=1,label=NameRS[irs-1])
                subfig[i,j].set_xlim((3.9,4.5))
                subfig[i,j].set_ylim((-0.05,0.25))
                subfig[i,j].set_xlabel(r'$M$ (GeV)',fontsize=fuente)
                subfig[i,j].set_ylabel(r'$\Gamma (GeV)$',fontsize=fuente)
                subfig[i,j].tick_params(direction='in',labelsize=fuente)
                subfig[i,j].legend(loc='upper right',ncol=1,frameon=True,fontsize=fuente)

    fig.savefig('polebs.pdf', bbox_inches='tight')

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
        N = input0[2]
        parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
        n0l, n1l, n2l, a00l, a11l, a22l, a01l, a02l, a12l = np.array_split(parameters_input,9)
        lmax = len(a00l)-1
        xsec = [ sigma_total(sarray[i],mphoton,mproton,mpsi,mproton,md,mlambdac,mdbar,mlambdac,N,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,lmax) for i in range(len(sarray))]
        fig = plt.figure()
        plt.xlim((Ebeamfroms(sth,mproton),15));
        plt.plot(Earray,xsec,'-',lw=2,c=jpac_color[0],alpha=1,zorder=2)
        fig.savefig('sigmatot.pdf', bbox_inches='tight')
        storage_plot[0,:], storage_plot[1,:], storage_plot[2,:] = Earray, sarray, xsec
        np.savetxt('sigmatot.txt', storage_plot)

elif option=='totalbs':

    nplotpoints = 100
    fuente = 20; 
    sth = (mproton + mpsi + 0.001)**2

    if modelo=='scat3':
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

    if modelo=='scat3':
        Earray, sarray, tarray, xsec, xsec_dw68, xsec_up68, xsec_dw95, xsec_up95 = xsec_file[0,:], xsec_file[1,:], xsec_file[2,:], xsec_file[3,:], xsec_file[4,:], xsec_file[5,:], xsec_file[6,:], xsec_file[7,:]
    else:            
        xsec, xsec_dw68, xsec_up68, xsec_dw95, xsec_up95 = bs_total(xbs,sarray,mphoton,mproton,mpsi,mproton,md,mlambdac,mdbar,mlambdac)
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



