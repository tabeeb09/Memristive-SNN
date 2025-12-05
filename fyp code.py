#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-12-05T17:31:13.845Z
"""

# ### Part 1: HP Memristor Model ###


import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.differentiate import derivative

R_on = 10 #ohm
R_off = 420
mu = 1.6*10**-12
D = 3*10**(-9) #m
p = 1 # empirical constant
I0 = 1*10**(-3) #initial current
wfrqu = 2*np.pi*(500)
V_plus = 0.37
V_minus = -0.19
i0 = 1*10**-3
i_off = 1*10**-5
i_on = 1
v0 = 0.6 #inital voltage

# def V(t): 
#     return v0*np.sin(wfrqu*t)

def M(x): 
    return R_on*(x) + R_off*(1-x)  

# def I(t,x): 
#     return V(t)/M(x)


def f(x): 
    return 1 - ((2*x - 1)**(2*1))

# def wode(t , x): 
    
#     if V(t) > V_plus: 
#         return mu*(R_on/D**2)*(i_off/(I(t, x)-i0))*f(x)
    
#     if V(t) < V_minus: 
#         return mu*(R_on/D**2)*(I(t, x)/(i_on))*f(x)
    
#     else: 
#         return 0

tval = np.linspace(0, 0.009, 1000)
t = tval.reshape(1000,1)

y0 = [0.01]

# x = solve_ivp(wode, (0,0.009), y0, t_eval = tval)
# x_val = x.y[0]
# x_val = x_val.reshape(1000,1)

# i = I(t, x_val)
# v = V(t)
# m = M(x_val)
# g = 1/m

# plt.plot(v, i, 'b')
# plt.xlabel("Voltage (V)")
# plt.ylabel("Current (A)")
# plt.show()
# print('1')

# plt.plot(t, m, 'b')
# plt.xlabel("Time (s)")
# plt.ylabel("Memristance ($\Omega$)")
# plt.show()
# print('2')
# plt.plot(v, g, 'b')
# plt.xlabel("Voltage (V)")
# plt.ylabel("Conductance (S)")
# plt.show()
# print('3')
# plt.plot(t, g, 'b')
# plt.xlabel("Time (s)")
# plt.ylabel("Conductance (S)")
# #plt.xlim(0.00307, 0.00311)
# #plt.ylim(0.068, 0.075)
# plt.show()
# print('4')
# print(np.max(g))
# print(np.min(g))

# plt.plot(t , x_val*D)
# plt.xlabel('Time(s)')
# plt.ylabel('$\Delta\omega$')

# tau = 0.00425
# dt = t - tau

# dg = ((g - 0.07)/0.07)*100

# plt.plot(dt, dg)
# plt.xlim(-0.0012, 0.00086)

from scipy import signal

def vsq(t): 
    return 0.6*(signal.square(wfrqu*t))

def Isq(t,x): 
    return vsq(t)/M(x)

#plt.plot(t, vsq(t))


def wodesq(t , x): 
    
    if vsq(t) > V_plus: 
        return mu*(R_on/D**2)*(i_off/(Isq(t, x)-i0))*f(x)
    
    if vsq(t) < V_minus: 
        return mu*(R_on/D**2)*(Isq(t, x)/(i_on))*f(x)
    
    else: 
        return 0

xsq = solve_ivp(wodesq, (0,0.009), y0, t_eval = tval)

xs = xsq.y[0]
xs = xs.reshape(len(xs), 1)

msq = M(xs)
gsq = 1/msq

# plt.plot(t, gsq)
# plt.xlabel('Time / s')
# plt.ylabel('Conductance / S')
#plt.show()

dt = (2*np.pi) / wfrqu
dg = np.concatenate([[0], np.diff(np.array(gsq), axis = 0)[:,0]]) 
print(dg, gsq)

plt.plot(t, dg)
plt.xlabel('Time / s')
plt.ylabel('Conductance / S')
plt.show()

# plt.plot(t ,xs)
# plt.xlabel('kawabunga 2')
# plt.ylabel('$\Delta\omega$')

# isq = Isq(t, xs)
# vsq = vsq(t)

# fig, ax1 = plt.subplots()
# ax1.plot(t, isq, color='red', label = 'Current')
# ax1.set_ylabel('Current / A', color='red')
# ax1.tick_params(axis='y', labelcolor='red')

# ax2 = ax1.twinx()
# ax2.plot(t, vsq, 'b', label = 'Voltage')
# ax2.set_ylabel('Voltage / V', color='b')
# ax2.tick_params(axis='y', labelcolor='b')
# plt.legend()
# #plt.show()
# print('5')

# #plt.xlim(0.0048, 0.0052)

# ap = 0.09 
# am = 0.04
# taup = 0.0003
# taum = 0.0002 

# dt = np.linspace(-0.005, 0.005, 1000)

# def stdp(dt): 
#     dt = np.array(dt)
#     dg = np.zeros_like(dt)

#     dg = np.where(dt>0, ap*np.exp(-dt/taup), dg)
#     dg = np.where(dt<0, -am*np.exp(dt/taum), dg)

#     return dg

# plt.plot(dt, stdp(dt))



# # ### Part 1.2: Landau–Khalatnikov (L–K) ferroelectric model ###


# alpha = -1
# beta = 1 
# gamma = 0.1
# G0 = np.min(g)
# k = 1*10**-6
# P = [0]
# max_g = np.max(g)
# min_g = np.min(g)

# def e(t): 
#     return V(t)/D

# # $ \frac{dp}{dt} = -\gamma(2\alpha p + 4\beta p^3 - e(t))$
# # 
# # p = polarisation 
# # 
# # e(t) = electric field 
# # 
# # gamma, beta and alpha are abirtary constants


# def dp(t, p): 
#     return -gamma * (2*alpha*p + 4*beta*p**3 - e(t))
    

# p = solve_ivp(dp, (0,0.009), P, t_eval = tval)
# pval = p.y[0]
# pval = pval.reshape(1000,1)

# pmin = np.min(pval)
# pmax = np.max(pval)

# G = min_g + ((max_g - min_g)*((pval + pmax)/(2*pmax)))

# plt.plot(t, G)
# #plt.xlim(0.001, 0.004)
# plt.ylabel('Conductance (S)')
# plt.xlabel('Time (s)')


# plt.plot(v, G)
# plt.xlabel('Volatge (V)')
# plt.ylabel('Conductance (S)')

# # $V_{th+} = +0.15V$
# # 
# # $V_{th-} = -0.21V$


# I = v*G
# plt.plot(v, I)
# plt.xlim(-0.3, 0.3)

# plt.plot(G,e(t))

# def esq(t): 
#     return vsq(t)/D

# def dpsq(t, p): 
#     return -gamma * (2*alpha*p + 4*beta*p**3 - esq(t))
    

# psq = solve_ivp

# # ### Part 2: STDP Model ###


# tau = 0.0022 - 0.0015
# dt = np.linspace(-0.02, 0.02, 10000)

# def stdp(dt): 
#     dt = np.array(dt)
#     dw = np.zeros_like(dt)

#     dw = np.where(dt>0, 0.01*np.exp(-dt/tau), dw)
#     dw = np.where(dt<0, -0.01*np.exp(dt/tau), dw)

#     return dw

# dtv = dt.reshape(10000,1)

# plt.plot(dtv, stdp(dtv))
# plt.axhline(0, ls='--', color='black')

# dgd = ((0.002 - 0.07)/0.07)*100
# dgp = ((0.0824 - 0.07)/0.07)*100

# def stdp(dt): 
#     dt = np.array(dt)
#     dw = np.zeros_like(dt)

#     dw = np.where(dt>0, dgp*np.exp(-dt/tau), dw)
#     dw = np.where(dt<0, dgd*np.exp(dt/tau), dw)

#     return dw

# plt.plot(dt, stdp(dt))
# plt.axhline(0, ls='--', color='black')
# plt.ylabel('$\Delta G$')
# plt.xlabel('$\Delta t$')
# #plt.xlim(-0.005, 0.005)
# plt.show()
# print('6')

# V_1 = 0 #initial voltage 
# V_2 = 0.6 #max voltage 
# tdr = 0.001 #time delay before rising 
# rt = 0.01 #rise time 
# tdf = 0.005 #time before falling 
# ft = 0.05 # fall time 

# def expv(t): 
#     t = np.array(t)
#     ev = np.zeros_like(t)

#     ev = np.where((tdr<t) & (t<=tdf) , V_1 + ((V_2 - V_1)*(1 - np.exp(-(t-tdr)/rt))), ev)
#     ev = np.where(t>tdf, V_2 - (V_2 - V_1)*((np.exp(-(t-tdf)/ft)) - (np.exp(-(t-tdr)/rt))), ev)

#     return ev

# tt = np.linspace(0, 1, 10000)
# plt.plot(tt, expv(tt))

# # $\tau \frac{dV}{dt} = E_l - V + R_{m} I_{e}$


# t_m = 0.02 #membrane time constant
# i_e =  0.06 #input current 
# e_l = -2 #resting potential of neuron 
# r_m = 100 # membrane resistance 

# def dv(t, v): 
#     return (e_l - v + (r_m*i_e))/t_m

# tnew = np.linspace(0, 0.05, 1000)
# tvnew = tnew.reshape(1000,1)

# V = np.zeros_like(tnew)

# for i in range(1, len(tnew)): 
#     V[i] = V[i-1] + 0.0001*((-V[i-1]+(r_m*i_e))/t_m)
#     if V[i] >= 2: 
#         V[i] = -2
# v_new = V.reshape(1000, 1)

# plt.plot(tvnew, v_new)
# plt.xlabel('Time / s')
# plt.ylabel('Voltage / V')

# def wnew(t , x): 
    
#     if v_new(t) > V_plus: 
#         return mu*(R_on/D**2)*(i_off/(I(t, x)-i0))*f(x)
    
#     if v_new(t) < V_minus: 
#         return mu*(R_on/D**2)*(I(t, x)/(i_on))*f(x)
    
#     else: 
#         return 0

# def x(t, x): 
#     tnew = np.array(tnew)
#     xval = np.zeros_like(tnew)

#     np.where(v_new>V_plus, mu*(R_on/D**2)*(i_off/(I(t, x)-i0))*f(x))

# # ### STDP 2.0 ###


# t_pre = 0.005
# t_post2 = 0.007
# t_post1 = 0.003
# sigma = 0.0003

# def V_pre(t): 
#     return 0.6*np.exp(-(t-t_pre)**2/(2*sigma**2))

# def V_post1(t): 
#     return -0.6*np.exp(-(t-t_post1)**2/(2*sigma**2))

# def V_post2(t): 
#     return -0.6*np.exp(-(t-t_post2)**2/(2*sigma**2))

# plt.plot(t, V_pre(t))
# plt.plot(t, V_post1(t))
# plt.plot(t, V_post2(t))

# vpre = V_pre(t)
# vpost1 = V_post1(t)
# vpost2 = V_post2(t)

# def v_comb(t): 
#     return V_pre(t) + V_post1(t) + V_post2(t)


# dt = t - 0.005

# plt.plot(dt, v_comb(dt))
# plt.xlabel('t/ s')
# plt.ylabel('Voltage')
# plt.axhline(0, ls='--', color='black')
# plt.title('Model Pre and Post Synaptic Voltage')
# plt.show()
# print('7')
# def Istdp(t,x): 
#     return v_comb(t)/M(x)

# print(Istdp(0.002, 0.5))

# def xstdp(t , x): 
    
#     if v_comb(t) > V_plus: 
#         return mu*(R_on/D**2)*(i_off/(Istdp(t, x)-i0))*f(x)
    
#     if v_comb(t) < V_minus: 
#         return mu*(R_on/D**2)*(Istdp(t, x)/(i_on))*f(x)
    
#     else: 
#         return 0

# xstdp = solve_ivp(xstdp, (0,0.009), y0, t_eval = tval)

# xst = xstdp.y[0]
# xst = xst.reshape(1000,1)

# mst = M(xst)
# gst = 1/mst

# plt.plot(t, gst)