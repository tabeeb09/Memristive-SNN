import numpy as np
import matplotlib.pyplot as plt

a = 0.02 #time scale of the recovery variable, u 
b = 0.2 #the sensitivity of the recovery variable u to the subthreshold fluctuations of the membrane potential v
c = -0.065 #spike reset (mV)
d = 0.008 #spike reset of u 


R_oncpg = 10 #ohm
R_offcpg = 20 #20
mu = 1.6*10**-16 #avergage ion mobility
D = 3*10**(-9) #length of filament
p = 1 # empirical constant

tiz = np.linspace(0, 10000, 20000)
dtiz = tiz[1] - tiz[0]
tizp = tiz.reshape(len(tiz))

v1 = np.zeros(len(tiz))
v2 = np.zeros(len(tiz))
u1 = np.zeros(len(tiz))
u2 = np.zeros(len(tiz))
x12 = np.zeros(len(tiz))
x21 = np.zeros(len(tiz))
v3 = np.zeros(len(tiz))
u3 = np.zeros(len(tiz))
x13 = np.zeros(len(tiz))
x31 = np.zeros(len(tiz))
x23 = np.zeros(len(tiz))
x32 = np.zeros(len(tiz))

#0.68283381  0.28435412 -0.09823795 -0.59336632 -0.07234725 -0.53996954

x32[0] = 0.1
x23[0] = 0.15
x13[0] = 0.15
x31[0] = 0.25
x12[0] = 0.5
x21[0] = 0.5

v1[0] = c 
v2[0] = c + 0.010
v3[0] = c + 0.015
u3[0] = b*c 
u1[0] = b*c
u2[0] = b*c 


def Mcpg(x): 
    return R_oncpg*(x) + R_offcpg*(1-x)  

def Gcpg(x): 
    return 1/Mcpg(x)

def f(x): 
    return 1 - ((2*x - 1)**(2*p))

# 3 Neuron CPG with IZ Neurons 
for i in range(1, len(tiz)): 
    vmem12 = v2[i-1] - v1[i-1]
    vmem21 = v1[i-1] - v2[i-1]
    vmem13 = v3[i-1] - v1[i-1]
    vmem31 = v1[i-1] - v3[i-1]
    vmem32 = v2[i-1] - v3[i-1]
    vmem23 = v3[i-1] - v2[i-1]

    e12, e13, e21, e23, e31, e32 = 0.54161049, 1.13004338, 0.4012895, 0.11940817, 0.96985994, -0.38100493  #0.64408114, 0.20485587, 0.14070611, -0.55619965, -0.21213989, -0.25285697#0.68283381, 0.28435412, -0.09823795, -0.59336632, -0.07234725, -0.53996954 #-0.12132506, 0.55557333, -0.39366201, -0.20711794, 0.21399778, 0.44523043 
    Isyn1 = e21 * vmem21*Gcpg(x21[i-1]) + e31 * vmem31*Gcpg(x31[i-1])
    Isyn2 = e12 * vmem12*Gcpg(x12[i-1]) + e32 * vmem32*Gcpg(x32[i-1])
    Isyn3 = e13 * vmem13*Gcpg(x13[i-1]) + e23 * vmem23*Gcpg(x23[i-1])

    
    # Isyn1 = vmem21*Gcpg(x21[i-1]) + vmem31*Gcpg(x31[i-1])
    # Isyn2 = vmem12*Gcpg(x12[i-1]) + vmem32*Gcpg(x32[i-1])
    # Isyn3 = vmem13*Gcpg(x13[i-1]) + vmem23*Gcpg(x23[i-1])
    
    
    dx12 = mu*(R_oncpg/D**2)*Isyn2*f(x12[i-1])
    dx21 = mu*(R_oncpg/D**2)*Isyn1*f(x21[i-1])
    dx32 = mu*(R_oncpg/D**2)*Isyn2*f(x32[i-1])
    dx23 = mu*(R_oncpg/D**2)*Isyn3*f(x23[i-1])
    dx13 = mu*(R_oncpg/D**2)*Isyn3*f(x13[i-1])
    dx31 = mu*(R_oncpg/D**2)*Isyn1*f(x31[i-1])

    x12[i] = x12[i-1] + (dx12*dtiz)
    x21[i] = x21[i-1] + (dx21*dtiz)
    x13[i] = x13[i-1] + (dx13*dtiz)
    x31[i] = x31[i-1] + (dx31*dtiz)
    x32[i] = x32[i-1] + (dx32*dtiz)
    x23[i] = x23[i-1] + (dx23*dtiz)

    dv1 = 40*((v1[i-1])**2) + (5*v1[i-1]) + 0.140 - u1[i-1] + 0.015 + Isyn1
    dv2 = 40*((v2[i-1])**2) + (5*v2[i-1]) + 0.140 - u2[i-1] + 0.015 + Isyn2
    dv3 = 40*((v3[i-1])**2) + (5*v3[i-1]) + 0.140 - u3[i-1] + 0.015 + Isyn3
    du1 = a*(b*v1[i-1] - u1[i-1])
    du2 = a*(b*v2[i-1] - u2[i-1])
    du3 = a*(b*v3[i-1] - u3[i-1])

    v1[i] = v1[i-1] + (dv1*dtiz)
    v2[i] = v2[i-1] + (dv2*dtiz)
    v3[i] = v3[i-1] + (dv3*dtiz)
    u1[i] = u1[i-1] + (du1*dtiz)
    u2[i] = u2[i-1] + (du2*dtiz)
    u3[i] = u3[i-1] + (du3*dtiz)

    if v1[i] >= 0.030:
        v1[i-1] = 0.030 
        v1[i] = c
        u1[i] = u1[i] + d
    if v2[i] >= 0.030:
        v2[i-1] = 0.030 
        v2[i] = c 
        u2[i] = u2[i] + d
    if v3[i] >= 0.030:
        v3[i-1] = 0.030 
        v3[i] = c 
        u3[i] = u3[i] + d

plt.plot(tizp, v1, color ='blue', label='Neuron 1')
plt.plot(tizp, v2, color ='red', label='Neuron 2')
plt.plot(tizp, v3, color ='green', label='Neuron 3')
plt.xlabel('Time(ms)')
plt.ylabel('Voltage (V)')
#plt.xlim(25,60)
plt.title('Potentials against Time of 3 Neurons')
plt.legend()
plt.show()

plt.plot(tizp, u1, color ='blue', label='Neuron 1')
plt.plot(tizp, u2, color ='red', label='Neuron 2')
plt.plot(tizp, u3, color ='green', label='Neuron 3')
plt.xlabel('Time(ms)')
plt.ylabel('Recovery Variable')
plt.title('Recovery Variable of Neurons')
plt.legend()
plt.show()

plt.plot(tizp, x12, label='Neuron 1 -> 2')
plt.plot(tizp, x21, label='Neuron 2 -> 1')
plt.plot(tizp, x13, label='Neuron 1 -> 3')
plt.plot(tizp, x31, label='Neuron 3 -> 1')
plt.plot(tizp, x32, label='Neuron 3 -> 2')
plt.plot(tizp, x23, label='Neuron 2 -> 3')
plt.xlabel('Time(ms)')
plt.ylabel('State Variable ')
#plt.xlim(0,15)
plt.title('State Variables of Memristive Synapses')
plt.legend()
plt.show()


