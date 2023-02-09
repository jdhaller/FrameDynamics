# simulate the perfect echo

import numpy as np
from FrameDynamics.Frame import Frame

# initialize frame
frame = Frame(["I", "S"])

# setup couplings
interaction = frame.set_interaction("I", "S", "Jweak")
interaction2 = frame.set_interaction("I", "S", "Jstrong")

# setup offsets
off = 2000
offsets1 = np.linspace(-off, off, 31)
offsets2 = np.linspace(-off, off, 30)
frame.set_offset("I", offsets1)
frame.set_offset("S", offsets2)

# define pulse sequence and simulate
frame.pulse(["I", "S"], degree=90, amplitude=10**(4), phase=0)
frame.delay(0.005)
frame.pulse(["I", "S"], 180, 10**(4), 1)
frame.delay(0.005)
frame.pulse(["I", "S"], 90, 10**(4), 1)
frame.delay(0.005)
frame.pulse(["I", "S"], 180, 10**(4), 1)
frame.delay(0.005)
frame.pulse(["I", "S"], 90, 10**(4), 2)

if __name__ == "__main__":    # required in Windows for multiprocessing
    frame.start(MP=True, traject=True)

    # plotting for interaction #1
    frame.plot_H0_2D(interaction)
    frame.plot_traject(interaction)

    # plotting for interaction #2
    frame.plot_H0_2D(interaction2)
    frame.plot_traject(interaction2, offsets={"I": 0, "S": 300})

    # read results and bilinear trajectories as dict-like objects
    results = frame.get_results()
    traject = frame.get_traject()



