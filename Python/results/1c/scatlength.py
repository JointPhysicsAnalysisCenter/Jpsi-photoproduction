#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:52:43 2023

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
import scipy.integrate as integrate

###############################################################################
#   Input
###############################################################################

modelos  = ['1c','2c','3c','1cb','2cb','3cb']

if len(sys.argv)<2:
    print('Number of input parameters should be 2, input was ',len(sys.argv))
    print('Input was:',sys.argv)
    sys.exit('Exiting due to error in input')

modelo  = sys.argv[1]
bffinput = sys.argv[2]
xbs = np.loadtxt(bffinput)

if len(sys.argv)>3:
    bffinput2 = sys.argv[3]
    xbs2 = np.loadtxt(bffinput2)
    nbs2  = len(xbs2[:,0])
    idown68, iup68 = int(np.trunc(0.16*nbs2)),  int(np.trunc(0.84*nbs2))
    idown95, iup95 = int(np.trunc(0.05*nbs2)), int(np.trunc(0.95*nbs2))

nbs  = len(xbs[:,0])
idown68, iup68 = int(np.trunc(0.16*nbs)),  int(np.trunc(0.84*nbs))
idown95, iup95 = int(np.trunc(0.05*nbs)), int(np.trunc(0.95*nbs))

print('Model',modelo)
print('Input file',bffinput)

if len(sys.argv)>3:
    print('Second input file',bffinput2)

if modelo=='1cb' or modelo=='2cb' or modelo=='3cb' or modelo=='3cbn':
    nbs = 1; print('BFF fit')
else:
    print('BS fits',nbs)

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
gevfm = 5.068; # 1 GeV = 5.068 fm^{-1}
vmd = 0.0273;
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
#   Phase space
###############################################################################

complexity = 0.0000000000000000001

def PhaseSpace(si,m1,m2):
    s  = si + 1j*complexity
    st = (m1+m2)**2
    xi = 1 - st/s
    q2 = kallen(s,m1**2,m2**2)/s/4.;
    q  = np.sqrt(q2)
    rho  = 2.*q/np.sqrt(s)
    log0 = rho*np.log((xi-rho)/(xi+rho))
    log1 = xi*(m2-m1)/(m1+m2)*np.log(m2/m1)
    return -(log0 +log1)/16./np.pi/np.pi


###############################################################################
#   Auxiliary functions
###############################################################################

#   L=0 zeta
def Fdirect_1c(si,m1,m2,m3,m4,n0,a0,b0):
    s = si + 1j*complexity
    K = a0+b0*kallen(s,m3**2,m4**2)/s/4.
    return np.abs(n0/(1.+PhaseSpace(s,m3,m4)*K))

def Fdirect_2c(si,m1,m2,m3,m4,m5,m6,n0,n1,a00,a01,a11,b00,b11):
    G0, G1 = PhaseSpace(si,m3,m4), PhaseSpace(si,m5,m6);
    s = si + 1j*complexity
    q02, q12 = kallen(s,m3**2,m4**2)/s/4., kallen(s,m5**2,m6**2)/s/4.;
    K00, K01, K11 = a00 + b00*q02, a01,  a11 + b11*q12
    K012 = K01*K01
    DeltaK = K00*K11-K012
    D = (1.+G0*K00)*(1.+G1*K11)-G0*G1*K012
    N0, A00 = n0, (1+G1*K11)/D
    return np.abs(N0*A00)

def Fdirect_3c(si,m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12):
    s  = si + 1j*complexity
    G0, G1, G2 = PhaseSpace(s,m1,m2), PhaseSpace(s,m3,m4), PhaseSpace(s,m5,m6);
    K00, K11, K22 = a00, a11, a22;
    K01, K02, K12 = a01, a02, a12;
    K012, K022, K122 = K01*K01, K02*K02, K12*K12;
    K3 = K01*K02*K12
    N0, A00 = n0, (1.+G1*K11)*(1.+G2*K22)-G1*G2*K122 
    D = (1.+G0*K00)*(1.+G1*K11)*(1.+G2*K22) - G0*G1*K012 - G0*G2*K022 - G1*G2*K122 + G0*G1*G2*( 2.*K3 - K11*K022 - K00*K122 - K22*K012 );
    return np.abs(N0*A00/D);

def Findirect_1c(si,m1,m2,m3,m4,n0,a0,b0):
    return 0.

def Findirect_2c(si,m1,m2,m3,m4,m5,m6,n0,n1,a00,a01,a11,b00,b11):
    G0, G1 = PhaseSpace(si,m3,m4), PhaseSpace(si,m5,m6);
    s = si + 1j*complexity
    q02, q12 = kallen(s,m3**2,m4**2)/s/4., kallen(s,m5**2,m6**2)/s/4.;
    K00, K01, K11 = a00 + b00*q02, a01,  a11 + b11*q12
    K012 = K01*K01
    D = (1.+G0*K00)*(1.+G1*K11)-G0*G1*K012
    N1 = n1
    A01 = K01/D;
    return np.abs(N1*G1*A01)

def Findirect_3c(si,m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12):
    s  = si + 1j*complexity
    G0, G1, G2 = PhaseSpace(s,m1,m2), PhaseSpace(s,m3,m4), PhaseSpace(s,m5,m6);
    K00, K11, K22 = a00, a11, a22;
    K01, K02, K12 = a01, a02, a12;
    K012, K022, K122 = K01*K01, K02*K02, K12*K12;
    K3 = K01*K02*K12
    N1, N2 = n1, n2
    A01, A02 = K01*(1.+G2*K22)-G2*K02*K12, K02*(1.+G1*K11)-G1*K01*K12; 
    D = (1.+G0*K00)*(1.+G1*K11)*(1.+G2*K22) - G0*G1*K012 - G0*G2*K022 - G1*G2*K122 + G0*G1*G2*( 2.*K3 - K11*K022 - K00*K122 - K22*K012 );
    return np.abs((N1*G1*A01+N2*G2*A02))#/D)

#   L=0 full amplitude
def F00_1c(si,m1,m2,m3,m4,n0,a0,b0):
    s = si + 1j*complexity
    K = a0+b0*kallen(s,m3**2,m4**2)/s/4.
    return np.real(n0/(1.+PhaseSpace(s,m3,m4)*K))

def F00_2c(si,m1,m2,m3,m4,m5,m6,n0,n1,a00,a01,a11,b00,b11):
    G0, G1 = PhaseSpace(si,m3,m4), PhaseSpace(si,m5,m6);
    s = si + 1j*complexity
    q02, q12 = kallen(s,m3**2,m4**2)/s/4., kallen(s,m5**2,m6**2)/s/4.;
    K00, K01, K11 = a00 + b00*q02, a01,  a11 + b11*q12
    K012 = K01*K01
    DeltaK = K00*K11-K012
    D = (1.+G0*K00)*(1.+G1*K11)-G0*G1*K012
    N0, N1 = n0, n1
    A00, A01 = (K00+G1*DeltaK)/D, K01/D;
    return np.real(N0*(1.-G0*A00) - N1*G1*A01)

def F00_3c(si,m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12):
    s  = si + 1j*complexity
    G0, G1, G2 = PhaseSpace(s,m1,m2), PhaseSpace(s,m3,m4), PhaseSpace(s,m5,m6);
    K00, K11, K22 = a00, a11, a22;
    K01, K02, K12 = a01, a02, a12;
    K012, K022, K122 = K01*K01, K02*K02, K12*K12;
    K3 = K01*K02*K12
    N0, N1, N2 = n0, n1, n2
    A00, A01, A02 = (1.+G1*K11)*(1.+G2*K22)-G1*G2*K122, K01*(1.+G2*K22)-G2*K02*K12, K02*(1.+G1*K11)-G1*K01*K12; 
    D = (1.+G0*K00)*(1.+G1*K11)*(1.+G2*K22) - G0*G1*K012 - G0*G2*K022 - G1*G2*K122 + G0*G1*G2*( 2.*K3 - K11*K022 - K00*K122 - K22*K012 );
    return np.real((N0*A00-N1*G1*A01-N2*G2*A02)/D)

#   L=0 elastic amplitude
def T00_1c(si,m1,m2,m3,m4,n0,a0,b0):
    s = si + 1j*complexity
    K = a0+b0*kallen(s,m3**2,m4**2)/s/4.
    return np.real(K/(1.+PhaseSpace(si,m3,m4)*K))

def T00_2c(si,m1,m2,m3,m4,m5,m6,n0,n1,a00,a01,a11,b00,b11):
    G0, G1 = PhaseSpace(si,m3,m4), PhaseSpace(si,m5,m6);
    s = si + 1j*complexity
    q02, q12 = kallen(s,m3**2,m4**2)/s/4., kallen(s,m5**2,m6**2)/s/4.;
    K00, K01, K11 = a00 + b00*q02, a01,  a11 + b11*q12
    K012 = K01*K01
    DeltaK = K00*K11-K012
    D = (1.+G0*K00)*(1.+G1*K11)-G0*G1*K012
    A00 = (K00+G1*DeltaK)/D
    return np.real(A00)

def T00_3c(si,m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12):
    s  = si + 1j*complexity
    G0, G1, G2 = PhaseSpace(s,m1,m2), PhaseSpace(s,m3,m4), PhaseSpace(s,m5,m6);
    K00, K11, K22 = a00, a11, a22;
    K01, K02, K12 = a01, a02, a12;
    K012, K022, K122 = K01*K01, K02*K02, K12*K12;
    K3 = K01*K02*K12
    A00 = K00 - G1*K012 - G2*K022 + G1*K00*K11 - G1*G2*K022*K11 + 2.*G1*G2*K01*K02*K12 - G1*G2*K00*K122 + G2*K00*K22 - G1*G2*K012*K22 + G1*G2*K00*K11*K22
    D = (1.+G0*K00)*(1.+G1*K11)*(1.+G2*K22) - G0*G1*K012 - G0*G2*K022 - G1*G2*K122 + G0*G1*G2*( 2.*K3 - K11*K022 - K00*K122 - K22*K012 );
    return np.real(A00/D)

#   Zeta
def zeta_1c(m1,m2,m3,m4,n0,a0,b0):
    sqst = m3+m4
    sth = sqst**2
    fd = Fdirect_1c(sth,m1,m2,m3,m4,n0,a0,b0)
    fi = Findirect_1c(sth,m1,m2,m3,m4,n0,a0,b0)
    return fi/(fi+fd)

def zeta_2c(m1,m2,m3,m4,m5,m6,n0,n1,a00,a01,a11,b00,b11):
    sqst = m3+m4
    sth = sqst**2
    fd = Fdirect_2c(sth,m1,m2,m3,m4,m5,m6,n0,n1,a00,a01,a11,b00,b11)
    fi = Findirect_2c(sth,m1,m2,m3,m4,m5,m6,n0,n1,a00,a01,a11,b00,b11)
    return fi/(fi+fd)

def zeta_3c(m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12):
    sqst = m1+m2
    sth = sqst**2
    fd = Fdirect_3c(sth,m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12)
    fi = Findirect_3c(sth,m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12)
    return fi/(fi+fd)

#   Scattering length
def apsip_1c(m1,m2,m3,m4,n0,a0,b0):
    sqst = m3+m4
    sth = sqst**2
    return -T00_1c(sth,m1,m2,m3,m4,n0,a0,b0)/(8.*np.pi*sqst);

def apsip_2c(m1,m2,m3,m4,m5,m6,n0,n1,a00,a01,a11,b00,b11):
    sqst = m3+m4
    sth = sqst**2
    return -T00_2c(sth,m1,m2,m3,m4,m5,m6,n0,n1,a00,a01,a11,b00,b11)/(8.*np.pi*sqst);

def apsip_3c(m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12):
    sqst = m1+m2
    sth = sqst**2
    return -T00_3c(sth,m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12)/(8.*np.pi*sqst);

#   VMD ratio
def rvmd_1c(m1,m2,m3,m4,n0,a0,b0):
    sqst = m3+m4
    sth = sqst**2
    return np.abs(F00_1c(sth,m1,m2,m3,m4,n0,a0,b0)/T00_1c(sth,m1,m2,m3,m4,n0,a0,b0)/vmd);

def rvmd_2c(m1,m2,m3,m4,m5,m6,n0,n1,a00,a01,a11,b00,b11):
    sqst = m3+m4
    sth = sqst**2
    return np.abs(F00_2c(sth,m1,m2,m3,m4,m5,m6,n0,n1,a00,a01,a11,b00,b11)/T00_2c(sth,m1,m2,m3,m4,m5,m6,n0,n1,a00,a01,a11,b00,b11)/vmd);

def rvmd_3c(m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12):
    sqst = m1+m2
    sth = sqst**2
    return np.abs(F00_3c(sth,m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12)/T00_3c(sth,m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12)/vmd);

#   Interaction range
def intr_1c(m1,m2,m3,m4,n0,a0,b0,a001):
    sqst = m3+m4
    sth = sqst**2
    return np.sqrt(np.abs(a001/T00_1c(sth,m1,m2,m3,m4,n0,a0,b0)));

def intr_2c(m1,m2,m3,m4,m5,m6,n0,n1,a00,a01,a11,b00,b11,a001):
    sqst = m3+m4
    sth = sqst**2
    return np.sqrt(np.abs(a001/T00_2c(sth,m1,m2,m3,m4,m5,m6,n0,n1,a00,a01,a11,b00,b11)));

def intr_3c(m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12,a001):
    sqst = m1+m2
    sth = sqst**2
    return np.sqrt(np.abs(a001/T00_3c(sth,m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12)));


def F_t(modelo,s,t,m00,m01,m1,m2,m3,m4,m5,m6,input0):
    if modelo=='1c' or modelo=='1cb':
        parameters_input = np.array([ input0[i] for i in range(2,len(input0))])
        nl, al, bl = np.array_split(parameters_input,3)
        lmax = len(nl)-1
        dsdt = dsigmadt_sc(s,t,m00,m01,m1,m2,nl,al,bl,lmax)
    elif modelo=='2c' or modelo=='2cb':
        N = input0[2]
        parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
        n0l, n1l, a00l, a01l, a11l, b00l, b01l, b11l = np.array_split(parameters_input,8)
        lmax = len(n0l)-1
        dsdt = dsigmadt_2c(s,t,m00,m01,m1,m2,m5,m6,N,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l,lmax)
    elif modelo=='3c' or modelo=='3cb':
        N = input0[2]
        parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
        n0l, n1l, n2l, a00l, a11l, a22l, a01l, a02l, a12l = np.array_split(parameters_input,9)
        lmax = len(n0l)-1
        dsdt = dsigmadt_3c(s,t,m00,m01,m1,m2,m3,m4,m5,m6,N,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,lmax)
    return dsdt


def RVMDt0(modelo,m00,m01,m1,m2,m3,m4,m5,m6,input0):
    conv = 1.
    t = 0.
    sqst = m1+m2+ 0.000001
    s = sqst**2 
    dsdt = F_t(modelo,s,t,m00,m01,m1,m2,m3,m4,m5,m6,input0)
    if modelo=='1c' or modelo=='1cb':
        parameters_input = np.array([ input0[i] for i in range(2,len(input0))])
        nl, al, bl = np.array_split(parameters_input,3)
        n0 = nl[0] 
        a0, b0 = conv*al[0], conv*bl[0]
        t00 = T00_1c(s,m00,m01,m1,m2,n0,a0,b0) 
    elif modelo=='2c' or modelo=='2cb':
        N = input0[2]
        parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
        n0l, n1l, a00l, a01l, a11l, b00l, b01l, b11l = np.array_split(parameters_input,8)
        n0, n1 = n0l[0], n1l[0]
        a00, a01, a11, b00, b01, b11 = conv*a00l[0], conv*a01l[0], conv*a11l[0], conv*b00l[0], conv*b01l[0], conv*b11l[0]
        t00 = T00_2c(s,m00,m01,m1,m2,m5,m6,n0,n1,a00,a01,a11,b00,b11)
    elif modelo=='3c' or modelo=='3cb':
        N = input0[2]
        parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
        n0l, n1l, n2l, a00l, a11l, a22l, a01l, a02l, a12l = np.array_split(parameters_input,9)
        n0, n1, n2 = n0l[0], n1l[0], n2l[0]
        a00, a11, a22, a01, a02, a12 = conv*a00l[0], conv*a11l[0], conv*a22l[0], conv*a01l[0], conv*a02l[0], conv*a12l[0]
        t00 = T00_3c(s,m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12)
    else:
        sys.exit('Wrong model,',modelo,'is not in',modelos)
        
    A = np.sqrt(dsdt)
    B = t00
    return np.absolute(A/B/vmd);

###############################################################################
#  ds/dt
###############################################################################

#   1c
def dsigmadt_sc(s,t,m1,m2,m3,m4,nl,al,bl,lmax):
    amplitude = Bcal_1c(s,t,m1,m2,m3,m4,nl,al,bl,lmax)
    num = np.absolute(amplitude)**2
    return num;

def Bcal_1c(s,t,m1,m2,m3,m4,nl,al,bl,lmax):
    return np.sum([ BcalL_1c(s,t,l,m1,m2,m3,m4,nl[l],al[l],bl[l]) for l in range(lmax+1)])

def BcalL_1c(s,t,l,m1,m2,m3,m4,nl,al,bl):
    x = costhetafromt(s,t,m1,m2,m3,m4)
    Lpol = LegPol(l,x)
    return (2.*l+1.)*Lpol*Amp_1c(s,l,m1,m2,m3,m4,nl,al,bl); 

def Amp_1c(s,l,m1,m2,m3,m4,nl,al,bl):
    p2, q2 = kallen(s,m1**2,m2**2)/s/4., kallen(s,m3**2,m4**2)/s/4.;
    p, q = np.sqrt(p2), np.sqrt(q2);
    K = (q2**l)*(al+bl*q2);
    N = nl*(p*q)**l
    return N/(1.+PhaseSpace(s,m3,m4)*K);

#   2c
def dsigmadt_2c(s,t,m1,m2,m3,m4,m5,m6,N,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l,lmax):
    amplitude = Bcal_2c(s,t,m1,m2,m3,m4,m5,m6,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l,lmax)
    num = np.absolute(amplitude)**2
    return num*N;

def BcalL_2c(s,t,l,m1,m2,m3,m4,m5,m6,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l):
    x = costhetafromt(s,t,m1,m2,m3,m4)
    Lpol = LegPol(l,x)
    return (2.*l+1.)*Lpol*Amp_2c(s,l,m1,m2,m3,m4,m5,m6,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l); 

def Bcal_2c(s,t,m1,m2,m3,m4,m5,m6,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l,lmax):
    return np.sum([ BcalL_2c(s,t,l,m1,m2,m3,m4,m5,m6,n0l[l],n1l[l],a00l[l],a01l[l],a11l[l],b00l[l],b01l[l],b11l[l]) for l in range(lmax+1)])

def Amp_2c(si,l,m1,m2,m3,m4,m5,m6,n0l,n1l,a00l,a01l,a11l,b00l,b01l,b11l):
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

#   3c
def dsigmadt_3c(s,t,m00,m01,m1,m2,m3,m4,m5,m6,N,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,lmax):
    amplitude = Bcal_3c(s,t,m00,m01,m1,m2,m3,m4,m5,m6,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,lmax)
    num = np.absolute(amplitude)**2
    return num*N;

def Amp_3c(si,l,m00,m01,m1,m2,m3,m4,m5,m6,n0l,n1l,n2l,a00,a11,a22,a01,a02,a12):
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

def BcalL_3c(s,t,l,m00,m01,m1,m2,m3,m4,m5,m6,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l):
    x = costhetafromt(s,t,m00,m01,m1,m2)
    Lpol = LegPol(l,x)
    return (2.*l+1.)*Lpol*Amp_3c(s,l,m00,m01,m1,m2,m3,m4,m5,m6,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l);

def Bcal_3c(s,t,m00,m01,m1,m2,m3,m4,m5,m6,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,lmax):
    return np.sum([ BcalL_3c(s,t,l,m00,m01,m1,m2,m3,m4,m5,m6,n0l[l],n1l[l],n2l[l],a00l[l],a11l[l],a22l[l],a01l[l],a02l[l],a12l[l]) for l in range(lmax+1)])

#   Right masses

apsip, rvmd, intr, zeta, rvmd0 = np.zeros(nbs), np.zeros(nbs), np.zeros(nbs), np.ones(nbs),  np.zeros(nbs)
m00, m01, m1, m2, m3, m4, m5, m6 = mphoton, mproton, mpsi, mproton, md, mlambdac, mdbar, mlambdac
conv = 1.
    
for ibs in range(nbs):
    input0 = xbs[ibs,:]
    rvmd0[ibs] = RVMDt0(modelo,m00,m01,m1,m2,m3,m4,m5,m6,input0)
    if modelo=='1c' or modelo=='1cb':
        parameters_input = np.array([ input0[i] for i in range(2,len(input0))])
        nl, al, bl = np.array_split(parameters_input,3)
        n0 = nl[0] 
        a0, b0 = conv*al[0], conv*bl[0]
        a1 = conv*al[1]
        apsip[ibs] = apsip_1c(m00,m01,m1,m2,n0,a0,b0)
        rvmd[ibs]  = rvmd_1c(m00,m01,m1,m2,n0,a0,b0)
        intr[ibs]  = intr_1c(m00,m01,m1,m2,n0,a0,b0,a1)
        zeta[ibs]  = 1.-zeta_1c(m00,m01,m1,m2,n0,a0,b0)
    elif modelo=='2c' or modelo=='2cb':
        N = input0[2]
        parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
        n0l, n1l, a00l, a01l, a11l, b00l, b01l, b11l = np.array_split(parameters_input,8)
        n0, n1 = n0l[0], n1l[0]
        a00, a01, a11, b00, b01, b11 = conv*a00l[0], conv*a01l[0], conv*a11l[0], conv*b00l[0], conv*b01l[0], conv*b11l[0]
        a001 = conv*a00l[1]
        apsip[ibs] = apsip_2c(m00,m01,m1,m2,m5,m6,n0,n1,a00,a01,a11,b00,b11)
        rvmd[ibs]  = rvmd_2c(m00,m01,m1,m2,m5,m6,n0,n1,a00,a01,a11,b00,b11)
        intr[ibs]  = intr_2c(m00,m01,m1,m2,m5,m6,n0,n1,a00,a01,a11,b00,b11,a001)
        zeta[ibs]  = 1.-zeta_2c(m00,m01,m1,m2,m5,m6,n0,n1,a00,a01,a11,b00,b11)
    elif modelo=='3c' or modelo=='3cb':
        N = input0[2]
        parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
        n0l, n1l, n2l, a00l, a11l, a22l, a01l, a02l, a12l = np.array_split(parameters_input,9)
        n0, n1, n2 = n0l[0], n1l[0], n2l[0]
        a00, a11, a22, a01, a02, a12 = conv*a00l[0], conv*a11l[0], conv*a22l[0], conv*a01l[0], conv*a02l[0], conv*a12l[0]
        a001 = conv*a00l[1]
        apsip[ibs] = apsip_3c(m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12)
        rvmd[ibs]  = rvmd_3c(m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12)
        intr[ibs]  = intr_3c(m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12,a001)
        zeta[ibs]  = 1.-zeta_3c(m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12)
    else:
        sys.exit('Wrong model,',modelo,'is not in',modelos)


def errorbar(vector,media):
    nbs = len(vector)
    i68 = int(np.trunc(0.68*nbs))
    i95 = int(np.trunc(0.9*nbs))
    r  = np.abs(vector- media)
    ra = r.argsort()
    vector_new , r_new= np.ones(nbs), np.zeros(nbs)
    for i in range(nbs):
        vector_new[i] = vector[ra[i]]
        r_new[i] = r[ra[i]]
    dw68, dw95 = media, media
    for i in range(i95):
        test = vector_new[i]
        if i<i68:
            if test<media:
                dw68, dw95 = test, test
            else:   
                up68, up95 = test, test
        else:
            if test<media:
                dw95 = test
            else:   
                up95 = test
    return dw68, up68, dw95, up95

def errorrange(vector):
    nbs = len(vector)
    id68, iu68 = int(np.trunc(0.16*nbs)),  int(np.trunc(0.84*nbs))
    id95, iu95 = int(np.trunc(0.05*nbs)), int(np.trunc(0.95*nbs))
    dw68, up68, dw95, up95 = vector[id68],vector[iu68],vector[id95],vector[iu95]
    return dw68, up68, dw95, up95

if modelo=='1c' or modelo=='2c' or modelo=='3c':
    apsip_sorted, rvmd_sorted, intr_sorted, zeta_sorted = np.sort(apsip)/gevfm, np.sort(rvmd), np.sort(intr)/gevfm , np.sort(zeta)  
    rvmd0_sorted = np.sort(rvmd0)
 
#    print(rvmd_sorted)
#    print(rvmd0_sorted)

    '''
    print('                    ')
    print('zeta=',np.mean(zeta),zeta_sorted[idown68],zeta_sorted[iup68],zeta_sorted[idown95],zeta_sorted[iup95])
    zeta_dw68 = zeta_sorted[idown68]
    zeta_up68 = zeta_sorted[iup68]
    zeta_dw95 = zeta_sorted[idown95]
    zeta_up95 = zeta_sorted[iup95]
    if modelo!='1c':
        new_zeta = np.delete(zeta_sorted, np.arange(iup68,len(zeta_sorted)))
        new_zeta = np.delete(new_zeta, np.arange(idown68))
        media, dev = np.mean(new_zeta), np.std(new_zeta)
        print('  std=',media,media-dev,media+dev,media-2.*dev,media+2.*dev)
        zeta_rec_dw68, zeta_rec_up68, zeta_rec_dw95, zeta_rec_up95 = errorbar(zeta,media)
        zeta_media = media
        print('  rec',zeta_media,zeta_rec_dw68, zeta_rec_up68, zeta_rec_dw95, zeta_rec_up95)
        np.savetxt('1C_zeta.txt', zeta_sorted)

    print('                    ')   
    print('a_psip=',np.mean(apsip)/gevfm,apsip_sorted[idown68],apsip_sorted[iup68],apsip_sorted[idown95],apsip_sorted[iup95])
    a_dw68 = apsip_sorted[idown68]
    a_up68 = apsip_sorted[iup68]
    a_dw95 = apsip_sorted[idown95]
    a_up95 = apsip_sorted[iup95]
    new_apsip = np.delete(apsip_sorted, np.arange(iup95,len(apsip_sorted)))
    new_apsip = np.delete(new_apsip, np.arange(idown95))
    media, dev = np.mean(new_apsip), np.std(new_apsip)
    print('  std=',media,media-dev,media+dev,media-2.*dev,media+2.*dev)
    a_rec_dw68, a_rec_up68, a_rec_dw95, a_rec_up95 = errorbar(apsip_sorted,media)
    a_media = media
    print('  rec',a_media,a_rec_dw68, a_rec_up68, a_rec_dw95, a_rec_up95)
    np.savetxt('1C_apsip.txt', apsip_sorted)

    print('                    ')
    print('R_vmd=',np.mean(rvmd),rvmd_sorted[idown68],rvmd_sorted[iup68],rvmd_sorted[idown95],rvmd_sorted[iup95])
    r_dw68 = rvmd_sorted[idown68]
    r_up68 = rvmd_sorted[iup68]
    r_dw95 = rvmd_sorted[idown95]
    r_up95 = rvmd_sorted[iup95]
    new_rvmd = np.delete(rvmd_sorted, np.arange(iup95,len(rvmd_sorted)))
    new_rvmd = np.delete(new_rvmd, np.arange(idown95))
    media, dev = np.mean(new_rvmd), np.std(new_rvmd)
    print('  std=',media,media-dev,media+dev,media-2.*dev,media+2.*dev)
    r_rec_dw68, r_rec_up68, r_rec_dw95, r_rec_up95 = errorbar(rvmd,media)
    r_media = media
    print('  rec',r_media,r_rec_dw68, r_rec_up68, r_rec_dw95, r_rec_up95)
    np.savetxt('1C_rvmd.txt', rvmd_sorted)

    print('                    ')
    print('r_eff=',np.mean(intr)/gevfm,intr_sorted[idown68],intr_sorted[iup68],intr_sorted[idown95],intr_sorted[iup95])
    ir_dw68 = intr_sorted[idown68]
    ir_up68 = intr_sorted[iup68]
    ir_dw95 = intr_sorted[idown95]
    ir_up95 = intr_sorted[iup95]
    new_intr = np.delete(intr_sorted, np.arange(iup95,len(intr_sorted)))
    new_intr = np.delete(new_intr, np.arange(idown95))
    media, dev = np.mean(new_intr), np.std(new_intr)
    print('  std=',media,media-dev,media+dev,media-2.*dev,media+2.*dev)
    ir_rec_dw68, ir_rec_up68, ir_rec_dw95, ir_rec_up95 = errorbar(intr_sorted,media)
    ir_media = media
    print('  rec',ir_media,ir_rec_dw68, ir_rec_up68, ir_rec_dw95, ir_rec_up95)
    np.savetxt('1C_intrange.txt', intr_sorted)
    '''

    '''
    for i in range(len(apsip)):
      if apsip[i]<0:
          print(i,apsip[i]/gevfm)
          for ibs in range(nbs):
              if ibs==i:
                  input0 = xbs[ibs,:]
                  N = input0[2]
                  parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
                  n0l, n1l, a00l, a01l, a11l, b00l, b01l, b11l = np.array_split(parameters_input,8)
                  n0, n1 = n0l[0], n1l[0]
                  a00, a01, a11, b00, b01, b11 = conv*a00l[0], conv*a01l[0], conv*a11l[0], conv*b00l[0], conv*b01l[0], conv*b11l[0]
                  print(ibs,n0, n1,a00, a01, a11, b00, b01, b11)
   '''


else:
    print('zeta=',zeta[0])
    print('a_psip=',apsip[0]/gevfm)
    print('R_vmd=',rvmd[0])
    print('R_vmd0=',rvmd0[0])
    print('r_eff=',intr[0]/gevfm)

#   Wrong masses

'''
if len(sys.argv)>3:
    md, mdbar, mlambdac = 1.86965, 2.00685, 2.28646; 
    conv = 16*np.pi
    if bffinput2=='pcbs_3cr_old.txt':
        conv = 1.
    apsip2, rvmd2, intr2, zeta2 = np.zeros(nbs2), np.zeros(nbs2), np.zeros(nbs2), np.ones(nbs2)
    m00, m01, m1, m2, m3, m4, m5, m6 = mphoton, mproton, mpsi, mproton, md, mlambdac, mdbar, mlambdac
    for ibs in range(nbs2):
        input0 = xbs2[ibs,:]
        if modelo=='1c' or modelo=='1cb':
            parameters_input = np.array([ input0[i] for i in range(2,len(input0))])
            nl, al, bl = np.array_split(parameters_input,3)
            n0 = nl[0] 
            a0, b0 = conv*al[0], conv*bl[0]
            a1 = conv*al[1]
            apsip2[ibs] = apsip_1c(m00,m01,m1,m2,n0,a0,b0)
            rvmd2[ibs]  = rvmd_1c(m00,m01,m1,m2,n0,a0,b0)
            intr2[ibs]  = intr_1c(m00,m01,m1,m2,n0,a0,b0,a1)
            zeta2[ibs]  = 1.-zeta_1c(m00,m01,m1,m2,n0,a0,b0)
        elif modelo=='2c' or modelo=='2cb':
            N = input0[2]
            parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
            n0l, n1l, a00l, a01l, a11l, b00l, b01l, b11l = np.array_split(parameters_input,8)
            n0, n1 = n0l[0], n1l[0]
            a00, a01, a11, b00, b01, b11 = conv*a00l[0], conv*a01l[0], conv*a11l[0], conv*b00l[0], conv*b01l[0], conv*b11l[0]
            a001 = conv*a00l[1]
            apsip2[ibs] = apsip_2c(m00,m01,m1,m2,m5,m6,n0,n1,a00,a01,a11,b00,b11)
            rvmd2[ibs]  = rvmd_2c(m00,m01,m1,m2,m5,m6,n0,n1,a00,a01,a11,b00,b11)
            intr2[ibs]  = intr_2c(m00,m01,m1,m2,m5,m6,n0,n1,a00,a01,a11,b00,b11,a001)
            zeta2[ibs]  = 1.-zeta_2c(m00,m01,m1,m2,m5,m6,n0,n1,a00,a01,a11,b00,b11)
        elif modelo=='3c' or modelo=='3cb':
            N = input0[2]
            parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
            n0l, n1l, n2l, a00l, a11l, a22l, a01l, a02l, a12l = np.array_split(parameters_input,9)
            n0, n1, n2 = n0l[0], n1l[0], n2l[0]
            a00, a11, a22, a01, a02, a12 = conv*a00l[0], conv*a11l[0], conv*a22l[0], conv*a01l[0], conv*a02l[0], conv*a12l[0]
            a001 = conv*a00l[1]
            apsip2[ibs] = apsip_3c(m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12)
            rvmd2[ibs]  = rvmd_3c(m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12)
            intr2[ibs]  = intr_3c(m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12,a001)
            zeta2[ibs]  = 1.-zeta_3c(m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12)
        else:
            sys.exit('Wrong model,',modelo,'is not in',modelos)

    print('                    ')
    print('                    ')
    print('                    ')
    print('Wrong masses')
    if modelo=='1c' or modelo=='2c' or modelo=='3c':
        apsip2_sorted, rvmd2_sorted, intr2_sorted, zeta2_sorted = np.sort(apsip2)/gevfm, np.sort(rvmd2), np.sort(intr2)/gevfm , np.sort(zeta2)  
        print('zeta=',np.mean(zeta2),zeta2_sorted[idown68],zeta2_sorted[iup68],zeta2_sorted[idown95],zeta2_sorted[iup95])
        print('a_psip=',np.mean(apsip2)/gevfm,apsip2_sorted[idown68],apsip2_sorted[iup68],apsip2_sorted[idown95],apsip2_sorted[iup95])
        print('R_vmd=',np.mean(rvmd2),rvmd2_sorted[idown68],rvmd2_sorted[iup68],rvmd2_sorted[idown95],rvmd2_sorted[iup95])
        print('r_eff=',np.mean(intr2)/gevfm,intr2_sorted[idown68],intr2_sorted[iup68],intr2_sorted[idown95],intr2_sorted[iup95])
    else:    
        print('zeta=',zeta2[0])
        print('a_psip=',apsip2[0]/gevfm)
        print('R_vmd=',rvmd2[0])
        print('r_eff=',intr2[0]/gevfm)
'''

def iterative_cut(vector):
    again = True
    while (again):
        length_vector = len(vector) 
        media, dev = np.mean(vector), np.std(vector)
        four_sigma = 4.*dev
        dw, up = media - four_sigma, media + four_sigma
        vector = vector[(vector >=dw) & (vector <= up)]
        if len(vector)==length_vector:
            again = False
    return vector

#   Iterative cut

def zeta_inv(zeta):
    return 1./(1.+np.exp(zeta))

if modelo=='1c' or modelo=='2c' or modelo=='3c':
    print('                    ')
    print('                    ')
    print('                    ')
    print('Iterative estimations with correct masses')

    if modelo!='1c':
        zeta_log_sorted = np.log(1./zeta_sorted-1.)
        b = iterative_cut(zeta_log_sorted)
        media = np.mean(b)
        np.savetxt('1C_zeta.txt',zeta_inv(b))
        dw68, up68, dw95, up95 = errorrange(b)#errorbar(b,media)
        print('zeta=',10000-len(b),np.around(zeta_inv(media), decimals=4),np.around(zeta_inv(up68), decimals=4),np.around(zeta_inv(dw68), decimals=4), np.around(zeta_inv(up95), decimals=4), np.around(zeta_inv(dw95), decimals=4))
    b = iterative_cut(apsip_sorted)
    np.savetxt('1C_a.txt',b)
    media = np.mean(b)
    dw68, up68, dw95, up95 = errorrange(b)#errorbar(b,media)
    print('a_psip=',10000-len(b),np.around(media,decimals=4),np.around(dw68,decimals=4),np.around(up68,decimals=4),np.around(dw95,decimals=4), np.around(up95,decimals=4)   )
    rvmd_log_sorted = np.log(rvmd_sorted)
    b = iterative_cut(rvmd_log_sorted )
    np.savetxt('1C_rvmd.txt',np.exp(b))
    media = np.mean(b)
    dw68, up68, dw95, up95 = errorrange(b)#errorbar(b,media)
    print('R_VMD=',10000-len(b),np.around(np.exp(media),decimals=4),np.around(np.exp(dw68),decimals=4),np.around(np.exp(up68),decimals=4),np.around(np.exp(dw95),decimals=4), np.around(np.exp(up95),decimals=4)   )
    b = iterative_cut(intr_sorted)
    np.savetxt('1C_ir.txt',b)
    media = np.mean(b)
    dw68, up68, dw95, up95 = errorrange(b)#errorbar(b,media)
    print('r_eff=',10000-len(b),np.around(media,decimals=4),np.around(dw68,decimals=4),np.around(up68,decimals=4),np.around(dw95,decimals=4), np.around(up95,decimals=4)   )
    rvmd0_log_sorted = np.log(rvmd0_sorted)
    b = iterative_cut(rvmd0_log_sorted )
    np.savetxt('1C_rvmd0.txt',np.exp(b))
    media = np.mean(b)
    dw68, up68, dw95, up95 = errorrange(b)#errorbar(b,media)
    print('R0_VMD=',10000-len(b),np.around(np.exp(media),decimals=4),np.around(np.exp(dw68),decimals=4),np.around(np.exp(up68),decimals=4),np.around(np.exp(dw95),decimals=4), np.around(np.exp(up95),decimals=4)   )
    print('                    ')
    print('                    ')
    print('                    ')


'''

#   Plot histograms  

if modelo=='1c' or modelo=='2c' or modelo=='3c':
    
    print('                    ')
    print('                    ')
    print('                    ')
    print('Peak estimations with correct masses')
    print('                    ')

    if modelo!='1c':
        #zeta    
        fig, subfig = plt.subplots(2,figsize=(8,8))
        npoints = 100
        dw, up = zeta_sorted[0], zeta_sorted[len(zeta_sorted)-1]
        if len(sys.argv)>3:
            dw, up = np.minimum(zeta_sorted[0],zeta2_sorted[0]), np.maximum(zeta_sorted[len(zeta_sorted)-1],zeta2_sorted[len(zeta2_sorted)-1])
            
        binwidth = np.abs(up+dw)/npoints
        bins = np.arange(zeta_sorted[0], zeta_sorted[len(zeta_sorted)-1], binwidth)
        subfig[0].set_title(r'$\zeta_{th}$')
        subfig[0].hist(zeta_sorted, bins=bins,color=jpac_blue,alpha=0.3,label=r'$\zeta$'+r'$\quad$'+modelo)
        if len(sys.argv)>3:
            subfig[0].hist(zeta2_sorted, bins=bins,color=jpac_red,alpha=0.3,label=r'$\zeta$ wrong')

        dw, up = zeta_sorted[idown95], zeta_sorted[iup95]
        if len(sys.argv)>3:
            dw, up = np.minimum(zeta_sorted[idown95],zeta2_sorted[idown95]), np.maximum(zeta_sorted[iup95],zeta2_sorted[iup95])
        binwidth = np.abs(up+dw)/npoints
        bins = np.arange(zeta_sorted[idown95], zeta_sorted[iup95], binwidth)
        subfig[1].set_title(r'$\zeta_{th}$')
        subfig[1].hist(zeta_sorted, bins=bins,color=jpac_blue,alpha=0.3,label=r'$\zeta$'+r'$\quad$'+modelo)
        if len(sys.argv)>3:
            subfig[1].hist(zeta2_sorted, bins=bins,color=jpac_red,alpha=0.3,label=r'$\zeta$ wrong')
        subfig[1].vlines(zeta_media,-120,0,colors=jpac_blue,linestyle='solid')        
        subfig[1].vlines([zeta_rec_dw68,zeta_rec_up68],-120,0,color=jpac_blue,linestyle='dashed')
        subfig[1].vlines([zeta_rec_dw95,zeta_rec_up95],-120,0,color=jpac_blue,linestyle='dashed')
        
        h = np.histogram(new_zeta, bins=bins) #zeta hack by using truncated new_zeta
        itemindex = np.where(h[0] == np.max(h[0]))
        peak, h0 = bins[itemindex][0], h[0][itemindex][0]
        peak_dw68, peak_up68, peak_dw95, peak_up95 = errorbar(new_zeta,peak)            
        print('zeta peak=',peak,peak_dw68, peak_up68, peak_dw95, peak_up95)
        subfig[1].vlines(peak,-120,0,colors=jpac_red,linestyle='solid')        
        subfig[1].vlines([peak_dw68,peak_up68],-120,0,colors=jpac_red,linestyle='dashed')
        subfig[1].vlines([peak_dw95,peak_up95],-120,0,colors=jpac_red,linestyle='dashed')
        
        subfig[1].vlines([zeta_dw68,zeta_up68],-120,0,colors=jpac_green,linestyle='dashed')
        subfig[1].vlines([zeta_dw95,zeta_up95],-120,0,colors=jpac_green,linestyle='dashed')
        subfig[1].legend(loc='upper right',ncol=1,frameon=True,fontsize=11)
        fig.savefig('fig_zeta.png', bbox_inches='tight')


    #a_psi   
    fig, subfig = plt.subplots(2,figsize=(8,8))
    npoints = 100
    dw, up = apsip_sorted[0], apsip_sorted[len(apsip_sorted)-1]
    if len(sys.argv)>3:
        dw, up = np.minimum(apsip_sorted[0],apsip2_sorted[0]), np.maximum(apsip_sorted[len(apsip_sorted)-1],apsip2_sorted[len(apsip2_sorted)-1])
    binwidth = np.abs(up+dw)/npoints
    bins = np.arange(apsip_sorted[0], apsip_sorted[len(zeta_sorted)-1], binwidth)
    subfig[0].set_title(r'$a_{\psi p}$')
    subfig[0].hist(apsip_sorted, bins=bins,color=jpac_blue,alpha=0.5,label=r'$a_{\psi p}$'+r'$\quad$'+modelo)
    if len(sys.argv)>3:
        subfig[0].hist(apsip2_sorted, bins=bins,color=jpac_red,alpha=0.5,label=r'$a_{\psi p}$ wrong')
    subfig[0].legend(loc='upper right',ncol=1,frameon=True,fontsize=11)

    dw, up = apsip_sorted[idown95], apsip_sorted[iup95]
    if len(sys.argv)>3:
        dw, up = np.minimum(apsip_sorted[idown95],apsip2_sorted[idown95]), np.maximum(apsip_sorted[iup95],apsip2_sorted[iup95])
    binwidth = np.abs(up+dw)/npoints
    bins = np.arange(apsip_sorted[idown95], apsip_sorted[iup95], binwidth)
    
    h = np.histogram(apsip_sorted, bins=bins)        
    itemindex = np.where(h[0] == np.max(h[0]))
    peak, h0 = bins[itemindex][0], h[0][itemindex][0]
    peak_dw68, peak_up68, peak_dw95, peak_up95 = errorbar(apsip_sorted,peak)
    print('a peak=',peak,peak_dw68, peak_up68, peak_dw95, peak_up95)

    subfig[1].set_title(r'$a_{\psi p}$')
    subfig[1].hist(apsip_sorted, bins=bins,color=jpac_blue,alpha=0.5,label=r'$a_{\psi p}$'+r'$\quad$'+modelo)
    if len(sys.argv)>3:
        subfig[1].hist(apsip2_sorted, bins=bins,color=jpac_red,alpha=0.5,label=r'$a_{\psi p}$ wrong')
    subfig[1].vlines(a_media,-120,0,colors=jpac_blue,linestyle='solid')
    subfig[1].vlines([a_rec_dw68,a_rec_up68],-120,0,color=jpac_blue,linestyle='dashed')
    subfig[1].vlines([a_rec_dw95,a_rec_up95],-120,0,color=jpac_blue,linestyle='dashed')

    subfig[1].vlines(peak,-120,0,colors=jpac_red,linestyle='solid')        
    subfig[1].vlines([peak_dw68,peak_up68],-120,0,colors=jpac_red,linestyle='dashed')
    subfig[1].vlines([peak_dw95,peak_up95],-120,0,colors=jpac_red,linestyle='dashed')

    subfig[1].vlines([a_dw68,a_up68],-120,0,colors=jpac_green,linestyle='dashed')
    subfig[1].vlines([a_dw95,a_up95],-120,0,colors=jpac_green,linestyle='dashed')
    subfig[1].legend(loc='upper right',ncol=1,frameon=True,fontsize=11)
    fig.savefig('fig_apsip.png', bbox_inches='tight')

    #R_vmd
    fig, subfig = plt.subplots(2,figsize=(8,8))
    npoints = 100    
    dw, up = rvmd_sorted[0], rvmd_sorted[len(zeta_sorted)-1]
    if len(sys.argv)>3:
        dw, up = np.minimum(rvmd_sorted[0],rvmd2_sorted[0]), np.maximum(rvmd_sorted[len(rvmd_sorted)-1],rvmd2_sorted[len(rvmd2_sorted)-1])    
    binwidth = np.abs(up+dw)/npoints
    bins = np.arange(rvmd_sorted[0], rvmd_sorted[len(zeta_sorted)-1], binwidth)
    subfig[0].set_title(r'$R_{VMD}$')
    subfig[0].hist(rvmd_sorted, bins=bins,color=jpac_blue,alpha=0.5,label=r'$R_{VMD}$'+r'$\quad$'+modelo)
    if len(sys.argv)>3:
        subfig[0].hist(rvmd2_sorted, bins=bins,color=jpac_red,alpha=0.5,label=r'$R_{VMD}$ wrong')
    subfig[0].legend(loc='upper right',ncol=1,frameon=True,fontsize=11)

    dw, up = rvmd_sorted[idown95], rvmd_sorted[iup95]
    if len(sys.argv)>3:
        dw, up = np.minimum(rvmd_sorted[idown95],rvmd2_sorted[idown95]), np.maximum(rvmd_sorted[iup95],rvmd2_sorted[iup95])
    binwidth = np.abs(up+dw)/npoints
    bins = np.arange(rvmd_sorted[idown95], rvmd_sorted[iup95], binwidth)
    subfig[1].set_title(r'$R_{VMD}$')
    subfig[1].hist(rvmd_sorted, bins=bins,color=jpac_blue,alpha=0.5,label=r'$R_{VMD}$'+r'$\quad$'+modelo)
    if len(sys.argv)>3:
        subfig[1].hist(rvmd2_sorted, bins=bins,color=jpac_red,alpha=0.5,label=r'$R_{VMD}$ wrong')
        
    subfig[1].vlines(r_media,-120,0,colors=jpac_blue,linestyle='solid')    
    subfig[1].vlines([r_rec_dw68,r_rec_up68],-120,0,color=jpac_blue,linestyle='dashed')
    subfig[1].vlines([r_rec_dw95,r_rec_up95],-120,0,color=jpac_blue,linestyle='dashed')
    
    h = np.histogram(rvmd_sorted, bins=bins)        
    itemindex = np.where(h[0] == np.max(h[0]))
    peak, h0 = bins[itemindex][0], h[0][itemindex][0]
    peak_dw68, peak_up68, peak_dw95, peak_up95 = errorbar(rvmd_sorted,peak)
    print('RVMD peak=',peak,peak_dw68, peak_up68, peak_dw95, peak_up95)
    subfig[1].vlines(peak,-120,0,colors=jpac_red,linestyle='solid')        
    subfig[1].vlines([peak_dw68,peak_up68],-120,0,colors=jpac_red,linestyle='dashed')
    subfig[1].vlines([peak_dw95,peak_up95],-120,0,colors=jpac_red,linestyle='dashed')

    subfig[1].vlines([r_dw68,r_up68],-120,0,colors=jpac_green,linestyle='dashed')
    subfig[1].vlines([r_dw95,r_up95],-120,0,colors=jpac_green,linestyle='dashed')
    subfig[1].legend(loc='upper right',ncol=1,frameon=True,fontsize=11)
    fig.savefig('fig_rvmd.png', bbox_inches='tight')

    #r_eff
    fig, subfig = plt.subplots(2,figsize=(8,8))
    npoints = 100
    dw, up = intr_sorted[0], intr_sorted[len(zeta_sorted)-1]
    if len(sys.argv)>3:
        dw, up = np.minimum(intr_sorted[0],intr2_sorted[0]), np.maximum(intr_sorted[len(intr_sorted)-1],intr2_sorted[len(intr2_sorted)-1])    
    binwidth = np.abs(up+dw)/npoints
    bins = np.arange(intr_sorted[0], intr_sorted[len(zeta_sorted)-1], binwidth)
    subfig[0].set_title(r'$r_{\psi p}$')
    subfig[0].hist(intr_sorted, bins=bins,color=jpac_blue,alpha=0.5,label=r'$r_{eff}$'+r'$\quad$'+modelo)    
    if len(sys.argv)>3:
        subfig[0].hist(intr2_sorted, bins=bins,color=jpac_red,alpha=0.5,label=r'$r_{eff}$ wrong')
    subfig[0].legend(loc='upper right',ncol=1,frameon=True,fontsize=11)

    dw, up = intr_sorted[idown95], intr_sorted[iup95]
    if len(sys.argv)>3:
        dw, up = np.minimum(intr_sorted[idown95],intr2_sorted[idown95]), np.maximum(intr_sorted[iup95],intr2_sorted[iup95])
    binwidth = np.abs(up+dw)/npoints
    bins = np.arange(intr_sorted[idown95], intr_sorted[iup95], binwidth)
    subfig[1].set_title(r'$r_{\psi p}$')
    subfig[1].hist(intr_sorted, bins=bins,color=jpac_blue,alpha=0.5,label=r'$r_{eff}$'+r'$\quad$'+modelo)
    if len(sys.argv)>3:
        subfig[1].hist(intr2_sorted, bins=bins,color=jpac_red,alpha=0.5,label=r'$r_{eff}$ wrong')
        
    subfig[1].vlines(ir_media,-120,0,colors=jpac_blue,linestyle='solid')        
    subfig[1].vlines([ir_rec_dw68,ir_rec_up68],-120,0,color=jpac_blue,linestyle='dashed')
    subfig[1].vlines([ir_rec_dw95,ir_rec_up95],-120,0,color=jpac_orange,linestyle='dashed')

    h = np.histogram(intr_sorted, bins=bins)        
    itemindex = np.where(h[0] == np.max(h[0]))
    peak, h0 = bins[itemindex][0], h[0][itemindex][0]
    peak_dw68, peak_up68, peak_dw95, peak_up95 = errorbar(intr_sorted,peak)
    print('reff peak=',peak,peak_dw68, peak_up68, peak_dw95, peak_up95)
    subfig[1].vlines(peak,-120,0,colors=jpac_red,linestyle='solid')        
    subfig[1].vlines([peak_dw68,peak_up68],-120,0,colors=jpac_red,linestyle='dashed')
    subfig[1].vlines([peak_dw95,peak_up95],-120,0,colors=jpac_red,linestyle='dashed')
    
    subfig[1].vlines([ir_dw68,ir_up68],-120,0,colors=jpac_green,linestyle='dashed')
    subfig[1].vlines([ir_dw95,ir_up95],-120,0,colors=jpac_green,linestyle='dashed')
    subfig[1].legend(loc='upper right',ncol=1,frameon=True,fontsize=11)    
    fig.savefig('fig_reff.png', bbox_inches='tight')

    print('                    ')
    print('                    ')
    print('                    ')


'''



