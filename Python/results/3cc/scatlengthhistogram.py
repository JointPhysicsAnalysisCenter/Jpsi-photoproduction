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

###############################################################################
#   Input
###############################################################################

xbs_f, xbs_r, xbs_nr  = np.loadtxt('pcbs_3c.txt'), np.loadtxt('pcbs_3cr.txt'), np.loadtxt('pcbs_3cnr.txt')
nbs_f, nbs_r, nbs_nr = len(xbs_f[:,0]), len(xbs_r[:,0]), len(xbs_nr[:,0])

idown68_f, iup68_f = int(np.trunc(0.16*nbs_f)),  int(np.trunc(0.84*nbs_f))
idown95_f, iup95_f = int(np.trunc(0.05*nbs_f)), int(np.trunc(0.95*nbs_f))

idown68_r, iup68_r = int(np.trunc(0.16*nbs_r)),  int(np.trunc(0.84*nbs_r))
idown95_r, iup95_r = int(np.trunc(0.05*nbs_r)), int(np.trunc(0.95*nbs_r))

idown68_nr, iup68_nr = int(np.trunc(0.16*nbs_nr)),  int(np.trunc(0.84*nbs_nr))
idown95_nr, iup95_nr = int(np.trunc(0.05*nbs_nr)), int(np.trunc(0.95*nbs_nr))

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

gevfm = 5.068; # 1 GeV = 5.068 fm^{-1}
vmd = 0.0273;
mproton, mpsi, mphoton = 0.938272, 3.0969160, 0.;
md, mdbar, mlambdac = 1.86484, 2.00685, 2.28646; 

m00, m01, m1, m2, m3, m4, m5, m6 = mphoton, mproton, mpsi, mproton, md, mlambdac, mdbar, mlambdac
conv = 1.

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
#  Error estimation
###############################################################################

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

def errorrange(vector):
    nbs = len(vector)
    id68, iu68 = int(np.trunc(0.16*nbs)),  int(np.trunc(0.84*nbs))
    id95, iu95 = int(np.trunc(0.05*nbs)), int(np.trunc(0.95*nbs))
    dw68, up68, dw95, up95 = vector[id68],vector[iu68],vector[id95],vector[iu95]
    return dw68, up68, dw95, up95

def zeta_inv(zeta):
    return 1./(1.+np.exp(zeta))


###############################################################################
#  Scattering length
###############################################################################

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
    return np.abs((N1*G1*A01+N2*G2*A02)/D)

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

def zeta_3c(m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12):
    sqst = m1+m2
    sth = sqst**2
    fd = Fdirect_3c(sth,m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12)
    fi = Findirect_3c(sth,m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12)
    return fi/(fi+fd)

def rvmd_3c(m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12):
    sqst = m1+m2
    sth = sqst**2
    return np.abs(F00_3c(sth,m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12)/T00_3c(sth,m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12)/vmd);

def apsip_3c(m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12):
    sqst = m1+m2
    sth = sqst**2
    return -T00_3c(sth,m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12)/(8.*np.pi*sqst);

def F_t(s,t,m00,m01,m1,m2,m3,m4,m5,m6,input0):
    N = input0[2]
    parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
    n0l, n1l, n2l, a00l, a11l, a22l, a01l, a02l, a12l = np.array_split(parameters_input,9)
    lmax = len(n0l)-1
    dsdt = dsigmadt_3c(s,t,m00,m01,m1,m2,m3,m4,m5,m6,N,n0l,n1l,n2l,a00l,a11l,a22l,a01l,a02l,a12l,lmax)
    return dsdt

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

def RVMDt0(m00,m01,m1,m2,m3,m4,m5,m6,input0):
    conv = 1.
    t = 0.
    sqst = m1+m2+ 0.000001
    s = sqst**2 
    dsdt = F_t(s,t,m00,m01,m1,m2,m3,m4,m5,m6,input0)
    N = input0[2]
    parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
    n0l, n1l, n2l, a00l, a11l, a22l, a01l, a02l, a12l = np.array_split(parameters_input,9)
    n0, n1, n2 = n0l[0], n1l[0], n2l[0]
    a00, a11, a22, a01, a02, a12 = conv*a00l[0], conv*a11l[0], conv*a22l[0], conv*a01l[0], conv*a02l[0], conv*a12l[0]
    t00 = T00_3c(s,m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12)
    A = np.sqrt(dsdt)
    B = t00
    return np.absolute(A/B/vmd);

###############################################################################
#  Calculation
###############################################################################
apsip_f, apsip_r, apsip_nr = np.zeros(nbs_f), np.zeros(nbs_r), np.zeros(nbs_nr)
zeta_f,  zeta_r,  zeta_nr  = np.zeros(nbs_f), np.zeros(nbs_r), np.zeros(nbs_nr)
rvmd_f,  rvmd_r,  rvmd_nr  = np.zeros(nbs_f), np.zeros(nbs_r), np.zeros(nbs_nr)
rvmd0_f, rvmd0_r, rvmd0_nr = np.zeros(nbs_f), np.zeros(nbs_r), np.zeros(nbs_nr)

for ibs in range(nbs_f):
    input0 = xbs_f[ibs,:]
    N = input0[2]
    parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
    n0l, n1l, n2l, a00l, a11l, a22l, a01l, a02l, a12l = np.array_split(parameters_input,9)
    n0, n1, n2 = n0l[0], n1l[0], n2l[0]
    a00, a11, a22, a01, a02, a12 = conv*a00l[0], conv*a11l[0], conv*a22l[0], conv*a01l[0], conv*a02l[0], conv*a12l[0]
    apsip_f[ibs] = apsip_3c(m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12)
    rvmd_f[ibs]  = rvmd_3c(m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12)
    zeta_f[ibs]  = 1.-zeta_3c(m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12)
    rvmd0_f[ibs]  = RVMDt0(m00,m01,m1,m2,m3,m4,m5,m6,input0)

for ibs in range(nbs_r):
    input0 = xbs_r[ibs,:]
    N = input0[2]
    parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
    n0l, n1l, n2l, a00l, a11l, a22l, a01l, a02l, a12l = np.array_split(parameters_input,9)
    n0, n1, n2 = n0l[0], n1l[0], n2l[0]
    a00, a11, a22, a01, a02, a12 = conv*a00l[0], conv*a11l[0], conv*a22l[0], conv*a01l[0], conv*a02l[0], conv*a12l[0]
    apsip_r[ibs] = apsip_3c(m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12)
    rvmd_r[ibs]  = rvmd_3c(m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12)
    zeta_r[ibs]  = 1.-zeta_3c(m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12)
    rvmd0_r[ibs]  = RVMDt0(m00,m01,m1,m2,m3,m4,m5,m6,input0)

for ibs in range(nbs_nr):
    input0 = xbs_nr[ibs,:]
    N = input0[2]
    parameters_input = np.array([ input0[i] for i in range(3,len(input0))])
    n0l, n1l, n2l, a00l, a11l, a22l, a01l, a02l, a12l = np.array_split(parameters_input,9)
    n0, n1, n2 = n0l[0], n1l[0], n2l[0]
    a00, a11, a22, a01, a02, a12 = conv*a00l[0], conv*a11l[0], conv*a22l[0], conv*a01l[0], conv*a02l[0], conv*a12l[0]
    apsip_nr[ibs] = apsip_3c(m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12)
    rvmd_nr[ibs]  = rvmd_3c(m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12)
    zeta_nr[ibs]  = 1.-zeta_3c(m00,m01,m1,m2,m3,m4,m5,m6,n0,n1,n2,a00,a11,a22,a01,a02,a12)
    rvmd0_nr[ibs]  = RVMDt0(m00,m01,m1,m2,m3,m4,m5,m6,input0)

#   Ordering,  90% confidence level estimation and histograms

#   Scattering length
apsip_f_sorted, apsip_r_sorted, apsip_nr_sorted = np.sort(apsip_f)/gevfm, np.sort(apsip_r)/gevfm, np.sort(apsip_nr)/gevfm

b = iterative_cut(apsip_f_sorted)
a_f_forhist = b
media = np.mean(b)
dw68, up68, dw95, up95 = errorrange(b)
dw68_f, up68_f, dw95_f, up95_f = dw68, up68, dw95, up95 
print('a_psip=',20000-len(b),np.around(media,decimals=4),np.around(dw68,decimals=4),np.around(up68,decimals=4),np.around(dw95,decimals=4), np.around(up95,decimals=4)   )

b = iterative_cut(apsip_r_sorted)
a_r_forhist = b
media = np.mean(b)
dw68, up68, dw95, up95 = errorrange(b)
dw68_r, up68_r, dw95_r, up95_r = dw68, up68, dw95, up95 
print('a_psip=',10000-len(b),np.around(media,decimals=4),np.around(dw68,decimals=4),np.around(up68,decimals=4),np.around(dw95,decimals=4), np.around(up95,decimals=4)   )

b = iterative_cut(apsip_nr_sorted)
a_nr_forhist = b
media = np.mean(b)
dw68, up68, dw95, up95 = errorrange(b)
dw68_nr, up68_nr, dw95_nr, up95_nr = dw68, up68, dw95, up95 
print('a_psip=',10000-len(b),np.around(media,decimals=4),np.around(dw68,decimals=4),np.around(up68,decimals=4),np.around(dw95,decimals=4), np.around(up95,decimals=4)   )

npoints = 100
dw, up = np.minimum(dw95_nr,dw95_r), np.maximum(up95_nr,up95_r)
binwidth = np.abs(dw-up)/npoints
bins = np.arange(dw, up, binwidth)

fig = plt.figure()
plt.xlim(-2.999, 0.5)
plt.ylim(0, 7)
plt.ylabel(r'Distribution density',fontsize=14)
plt.xlabel(r'$a_{\psi p}$ [fm]',fontsize=14)
plt.hist(a_nr_forhist, bins=bins,density=True,color=jpac_green,alpha=0.5,label='Non-resonant (3C-NR)')
plt.hist(a_r_forhist, bins=bins,density=True,color=jpac_orange,alpha=0.5,label='Resonant (3C-R)')
plt.tick_params(direction='in',labelsize=12)
plt.legend(loc='center',ncol=1,frameon=False,fontsize=12)
fig.savefig('scatlength_histogram.pdf', bbox_inches='tight')
fig.savefig('scatlength_histogram.png', bbox_inches='tight')


#   zeta
zeta_f_sorted, zeta_r_sorted, zeta_nr_sorted = np.sort(zeta_f), np.sort(zeta_r), np.sort(zeta_nr)

zeta_log_sorted = np.log(1./zeta_f_sorted-1.)
x = iterative_cut(zeta_log_sorted)
b = zeta_inv(zeta_log_sorted)
zeta_f_forhist = b
media = np.mean(b)
dw68, up68, dw95, up95 = errorrange(b)
dw68_f, up68_f, dw95_f, up95_f = dw68, up68, dw95, up95 
print('zeta=',20000-len(b),np.around(media,decimals=4),np.around(dw68,decimals=4),np.around(up68,decimals=4),np.around(dw95,decimals=4), np.around(up95,decimals=4)   )

zeta_log_sorted = np.log(1./zeta_r_sorted-1.)
x = iterative_cut(zeta_log_sorted)
b = zeta_inv(zeta_log_sorted)
zeta_r_forhist = b
media = np.mean(b)
dw68, up68, dw95, up95 = errorrange(b)
dw68_r, up68_r, dw95_r, up95_r = dw68, up68, dw95, up95 
print('zeta=',10000-len(b),np.around(media,decimals=4),np.around(dw68,decimals=4),np.around(up68,decimals=4),np.around(dw95,decimals=4), np.around(up95,decimals=4)   )

zeta_log_sorted = np.log(1./zeta_nr_sorted-1.)
x = iterative_cut(zeta_log_sorted)
b = zeta_inv(zeta_log_sorted)
zeta_nr_forhist = b
media = np.mean(b)
dw68, up68, dw95, up95 = errorrange(b)
dw68_nr, up68_nr, dw95_nr, up95_nr = dw68, up68, dw95, up95 
print('zeta=',10000-len(b),np.around(media,decimals=4),np.around(dw68,decimals=4),np.around(up68,decimals=4),np.around(dw95,decimals=4), np.around(up95,decimals=4)   )

npoints = 100
dw, up = np.minimum(dw95_nr,dw95_r), np.maximum(up95_nr,up95_r)
binwidth = np.abs(dw-up)/npoints
bins = np.arange(dw, up, binwidth)

fig = plt.figure()
#plt.xlim(-2.999, 0.5)
plt.ylabel(r'density',fontsize=14)
plt.xlabel(r'$\zeta_{th}$',fontsize=14)
plt.hist(zeta_nr_forhist, bins=bins,density=True,color=jpac_green ,alpha=0.3,label='3C-NR')
plt.hist(zeta_r_forhist, bins=bins,density=True,color=jpac_orange,alpha=0.3,label='3C-R')
plt.tick_params(direction='in',labelsize=12)
plt.legend(loc='upper left',ncol=1,frameon=False,fontsize=12)
fig.savefig('zeta_histogram.pdf', bbox_inches='tight')
fig.savefig('zeta_histogram.png', bbox_inches='tight')

#   rvmd
rvmd_f_sorted, rvmd_r_sorted, rvmd_nr_sorted = np.sort(rvmd_f), np.sort(rvmd_r), np.sort(rvmd_nr)

rvmd_log_sorted = np.log(rvmd_f_sorted)
x = iterative_cut(rvmd_log_sorted)
b = np.exp(x)
rvmd_f_forhist = b
media = np.mean(b)
dw68, up68, dw95, up95 = errorrange(b)
dw68_f, up68_f, dw95_f, up95_f = dw68, up68, dw95, up95 
print('RVMD=',20000-len(b),np.around(media,decimals=4),np.around(dw68,decimals=4),np.around(up68,decimals=4),np.around(dw95,decimals=4), np.around(up95,decimals=4)   )

rvmd_log_sorted = np.log(rvmd_r_sorted)
x = iterative_cut(rvmd_log_sorted)
b = np.exp(x)
rvmd_r_forhist = b
media = np.mean(b)
dw68, up68, dw95, up95 = errorrange(b)
dw68_r, up68_r, dw95_r, up95_r = dw68, up68, dw95, up95 
print('RVMD=',10000-len(b),np.around(media,decimals=4),np.around(dw68,decimals=4),np.around(up68,decimals=4),np.around(dw95,decimals=4), np.around(up95,decimals=4)   )

rvmd_log_sorted = np.log(rvmd_nr_sorted)
x = iterative_cut(rvmd_log_sorted)
b = np.exp(x)
rvmd_nr_forhist = b
media = np.mean(b)
dw68, up68, dw95, up95 = errorrange(b)
dw68_nr, up68_nr, dw95_nr, up95_nr = dw68, up68, dw95, up95 
print('RVMD=',10000-len(b),np.around(media,decimals=4),np.around(dw68,decimals=4),np.around(up68,decimals=4),np.around(dw95,decimals=4), np.around(up95,decimals=4)   )

npoints = 100
dw, up = np.minimum(dw95_nr,dw95_r), np.maximum(up95_nr,up95_r)
binwidth = np.abs(dw-up)/npoints
bins = np.arange(dw, up, binwidth)

bins = np.logspace(np.log10(dw), stop=np.log10(up), num=npoints)

fig = plt.figure()
plt.xscale('log')      
plt.ylabel(r'N',fontsize=14)
plt.xlabel(r'$R_{VMD}(\theta=0)$',fontsize=14)
plt.hist(rvmd_r_forhist, bins=bins,density=True,color=jpac_orange,alpha=0.3,label='3C-R')
plt.hist(rvmd_nr_forhist, bins=bins,density=True,color=jpac_green,alpha=0.3,label='3C-NR')
plt.tick_params(direction='in',labelsize=12)
plt.legend(loc='upper right',ncol=1,frameon=False,fontsize=12)
fig.savefig('rvmd_histogram.pdf', bbox_inches='tight')
fig.savefig('rvmd_histogram.png', bbox_inches='tight')

#   rvmd0
rvmd_f_sorted, rvmd_r_sorted, rvmd_nr_sorted = np.sort(rvmd0_f), np.sort(rvmd0_r), np.sort(rvmd0_nr)

rvmd_log_sorted = np.log(rvmd_f_sorted)
x = iterative_cut(rvmd_log_sorted)
b = np.exp(x)
rvmd_f_forhist = b
media = np.mean(b)
dw68, up68, dw95, up95 = errorrange(b)
dw68_f, up68_f, dw95_f, up95_f = dw68, up68, dw95, up95 
print('RVMD0=',20000-len(b),np.around(media,decimals=4),np.around(dw68,decimals=4),np.around(up68,decimals=4),np.around(dw95,decimals=4), np.around(up95,decimals=4)   )

rvmd_log_sorted = np.log(rvmd_r_sorted)
x = iterative_cut(rvmd_log_sorted)
b = np.exp(x)
rvmd_r_forhist = b
media = np.mean(b)
dw68, up68, dw95, up95 = errorrange(b)
dw68_r, up68_r, dw95_r, up95_r = dw68, up68, dw95, up95 
print('RVMD0=',10000-len(b),np.around(media,decimals=4),np.around(dw68,decimals=4),np.around(up68,decimals=4),np.around(dw95,decimals=4), np.around(up95,decimals=4)   )

rvmd_log_sorted = np.log(rvmd_nr_sorted)
x = iterative_cut(rvmd_log_sorted)
b = np.exp(x)
rvmd_nr_forhist = b
media = np.mean(b)
dw68, up68, dw95, up95 = errorrange(b)
dw68_nr, up68_nr, dw95_nr, up95_nr = dw68, up68, dw95, up95 
print('RVMD0=',10000-len(b),np.around(media,decimals=4),np.around(dw68,decimals=4),np.around(up68,decimals=4),np.around(dw95,decimals=4), np.around(up95,decimals=4)   )

npoints = 100
dw, up = np.minimum(dw95_nr,dw95_r), np.maximum(up95_nr,up95_r)
binwidth = np.abs(dw-up)/npoints
bins = np.arange(dw, up, binwidth)

bins = np.logspace(np.log10(dw), stop=np.log10(up), num=npoints)

fig = plt.figure()
plt.xscale('log')      
plt.ylabel(r'N',fontsize=14)
plt.xlabel(r'$R_{VMD}(t=0)$',fontsize=14)
plt.hist(rvmd_r_forhist, bins=bins,color=jpac_blue,alpha=0.3,label='3C-R')
plt.hist(rvmd_nr_forhist, bins=bins,color=jpac_red ,alpha=0.3,label='3C-NR')
plt.tick_params(direction='in',labelsize=12)
plt.legend(loc='upper right',ncol=1,frameon=False,fontsize=12)
fig.savefig('rvmd0_histogram.pdf', bbox_inches='tight')
fig.savefig('rvmd0_histogram.png', bbox_inches='tight')


'''
fig, subfig = plt.subplots(2,figsize=(3,3))

for i in np.arange(2):
    subfig[i].axis('off')

left, width = 0.1, 0.9; bottom, height = 0.1, 0.8 ;
rect_histy = [left, bottom, width, height]
subfig[0] = plt.axes(rect_histy)

subfig[0].set_xlim(-2.999, 0.5)
subfig[0].set_ylabel(r'$N$',fontsize=14)
subfig[0].set_xlabel(r'$a_{\psi p}$ (fm)',fontsize=14)
subfig[0].hist(a_r_forhist, bins=bins,color=jpac_blue,alpha=0.3,label='3C-R')
subfig[0].hist(a_nr_forhist, bins=bins,color=jpac_red ,alpha=0.3,label='3C-NR')
subfig[0].tick_params(direction='in',labelsize=12)
subfig[0].legend(loc='center left',ncol=1,frameon=False,fontsize=12)


left, width = 0.15, 0.2; bottom, height = 0.7, 0.2;
rect_histy = [left, bottom, width, height]
subfig[1] = plt.axes(rect_histy)
subfig[1].axis('off')
file_jpac_logo = "JPAC_logo_color-1.png"
jpac_logo = plt.imread(file_jpac_logo)
subfig[1].imshow(jpac_logo)

fig.savefig('scatlength_histogram.pdf', bbox_inches='tight')
'''





