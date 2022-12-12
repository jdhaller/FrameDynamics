# simulate the isotropic perfect echo 2

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
ph2 = [3,3,0,0,3,3,  1,1,2,2,1,1]
N = 2
tau = 0.005
phcorr = 4/np.pi * 1/(4*amp)

# define pulse sequence
for _ in range(N):
    frame.delay( (tau+phcorr) / 2 )
    frame.pulse(["I", "S"], 180, amp, ph2[j]); j+=1
    frame.delay( tau+phcorr )
    frame.pulse(["I", "S"], 180, amp, ph2[j]); j+=1
    frame.delay( tau/2 )
    frame.pulse(["I", "S"], 90, amp, ph1[i]); i+=1
    frame.delay( tau/2 )
    frame.pulse(["I", "S"], 180, amp, ph2[j]); j+=1
    frame.delay( tau+phcorr )
    frame.pulse(["I", "S"], 180, amp, ph2[j]); j+=1
    frame.delay( tau/2 )
    frame.pulse(["I", "S"], 90, amp, ph1[i]); i+=1
    frame.delay( tau/2 )
    frame.pulse(["I", "S"], 180, amp, ph2[j]); j+=1
    frame.delay( tau+phcorr )
    frame.pulse(["I", "S"], 180, amp, ph2[j]); j+=1
    frame.delay( (tau+phcorr)/2 )

i -= 1
j -= 1

for _ in range(N):
    frame.delay( (tau+phcorr) / 2 )
    frame.pulse(["I", "S"], 180, amp, ph2[j]+2); j-=1
    frame.delay( tau+phcorr )
    frame.pulse(["I", "S"], 180, amp, ph2[j]+2); j-=1
    frame.delay( tau/2 )
    frame.pulse(["I", "S"], 90, amp, ph1[i]+2); i-=1
    frame.delay( tau/2 )
    frame.pulse(["I", "S"], 180, amp, ph2[j]+2); j-=1
    frame.delay( tau+phcorr )
    frame.pulse(["I", "S"], 180, amp, ph2[j]+2); j-=1
    frame.delay( tau/2 )
    frame.pulse(["I", "S"], 90, amp, ph1[i]+2); i-=1
    frame.delay( tau/2 )
    frame.pulse(["I", "S"], 180, amp, ph2[j]+2); j-=1
    frame.delay( tau+phcorr )
    frame.pulse(["I", "S"], 180, amp, ph2[j]+2); j-=1
    frame.delay( (tau+phcorr)/2 )

# start simulation
frame.start(MP=False, traject=True)

# plotting 
frame.plot_traject(interaction)


