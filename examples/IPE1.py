# simulate the isotropic perfect echo 1

import numpy as np
from FrameDynamics.Frame import Frame

# initialize frame
frame = Frame(["I", "S"])

# setup couplings
interaction = frame.set_interaction("I", "S", "Jweak")

# setup pulse sequence parameters 
i, j = 0, 0
amp = 10**(3)
ph1 = [2,3,1,0,  2,1,3,0,  0,3,1,2,  0,1,3,2]
ph2 = [0,0,1,3,0,2,  3,3,0,0,3,3,  2,0,1,3,0,0,  3,3,0,0,3,3]
N = 4
tau = 0.005
phcorr = 2/np.pi * 1/(4*amp)

# define pulse sequence
for _ in range(N):
    frame.delay(tau + phcorr)
    frame.pulse(["I", "S"], 180, amp, ph2[j]); j+=1
    frame.delay(tau)
    frame.pulse(["I", "S"], 90, amp, ph1[i]); i+=1
    frame.delay(tau)
    frame.pulse(["I", "S"], 180, amp, ph2[j]); j+=1
    frame.delay(tau)
    frame.pulse(["I", "S"], 90, amp, ph1[i]); i+=1
    frame.delay(tau)
    frame.pulse(["I", "S"], 180, amp, ph2[j]); j+=1
    frame.delay(tau + phcorr)

i -= 1
j -= 1

for _ in range(N):
    frame.delay(tau + phcorr)
    frame.pulse(["I", "S"], 180, amp, ph2[j]+2); j-=1
    frame.delay(tau)
    frame.pulse(["I", "S"], 90, amp, ph1[i]+2); i-=1
    frame.delay(tau)
    frame.pulse(["I", "S"], 180, amp, ph2[j]+2); j-=1
    frame.delay(tau)
    frame.pulse(["I", "S"], 90, amp, ph1[i]+2); i-=1
    frame.delay(tau)
    frame.pulse(["I", "S"], 180, amp, ph2[j]+2); j-=1
    frame.delay(tau + phcorr)

# start simulation
frame.start(Traject=True)

# plotting 
frame.plot_traject(interaction)


