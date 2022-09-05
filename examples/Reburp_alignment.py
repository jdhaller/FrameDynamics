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
block1.shape(["I"], Reburp, 1000*10**(-6), 6264.8, 1)
block2.pulse(["S"], 180, 10**(5), 1)

# align Reburp ("I") and hard pulse ("S") and simulate 
frame.align(block1, block2, alignment="center")
frame.start(traject=True)

# plotting
frame.plot_H0_2D(interaction)
frame.plot_H0_1D(interaction, "S", offset=0.)
frame.plot_traject(interaction)


