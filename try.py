import numpy as np
tmax = 100
dt = 1
D=5
A=3.5
F=14
Va=0.2
sh=1/Va**2
sc=F/sh
t=np.arange(0, tmax, dt)
Ism = np.zeros (len(t))
p=int(round(D/dt))
pulse=A*np.ones(p)
instfreq=np.random.gamma(sh,sc)
ipi=tmax/instfreq
ip=int(round(ipi/dt))
i=ip

while i<len(t)and i+p<(len(t)):
    Ism[i:i+p]= pulse
    instfreq=np.random.gamma(sh,sc)
    ipi=tmax/instfreq
    ip=int(round(ipi/dt))
    i+=ip