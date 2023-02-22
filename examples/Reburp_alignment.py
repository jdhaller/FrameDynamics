# simulate coupling evolution during heteronuclear echo 
# where a Reburp is aligned with a hard 180° pulse.

import numpy as np
from FrameDynamics.Frame import Frame
from FrameDynamics.Block import Block

# initialize frame
frame = Frame(["I", "S"])

# setup interaction
interaction = frame.set_interaction("I", "S", "Jweak")

# setup offsets
off = 5000
offsetsI = np.linspace(-off, off, 61)
offsetsS = np.linspace(-off, off, 61)
frame.set_offset("I", offsetsI)
frame.set_offset("S", offsetsS)

# load pulse shape
Reburp = frame.load_shape("Reburp.1000")

# initialize pulse sequence blocks for alignment
block1 = Block(frame, ["I"])
block2 = Block(frame, ["S"])

# define a Reburp pulse on "I" and hard pulse on "S"
block1.shape(["I"], Reburp, length=1000*10**(-6), amplitude=6264.8, phase=1)
block2.pulse(["S"], degree=180, amplitude=10000, phase=1)

# align Reburp ("I") and hard pulse ("S") and simulate 
frame.align(block1, block2, alignment="center")

if __name__ == "__main__":    # required in Windows for multiprocessing
    frame.start(MP=True, traject=True)

    # plotting
    frame.plot_H0_2D(interaction)
    frame.plot_H0_1D(interaction, "S", offset=0.)
    frame.plot_traject(interaction)

    time, traject = frame.get_traject(interaction,
                                      offsets={"I": 0, "S": 300},
                                      operators=["1z", "zz"])
    average_Hamiltonian = frame.get_results(interaction,
                                            operators=["zz"])
