# simulate homonuclear coupling evolution during chirp pulse

import numpy as np
from FrameDynamics import Frame

# initialize frame
frame = Frame(["I", "J"])

# setup interaction
interaction = frame.set_interaction("I", "J", "Jstrong")

# setup offsets
off = 40000
offsetsI = np.linspace(-off, off, 101)
offsetsS = np.linspace(-off, off, 101)
frame.set_offset("I", offsetsI)
frame.set_offset("J", offsetsS)

# setup shaped pulse
Chirp = frame.load_shape("SmCrp_BW50_RF9403_720u")

# simulate chirp pulse
frame.shape(["I", "J"], Chirp, 720*10**(-6), 9403, 1) 
frame.start(Traject=True)

# plotting
frame.plot_AHT2D(interaction, save="Chirp_TF2D.png")
frame.plot_AHT1D(interaction, "J", offset=0., save="Chirp_TF1D.png")
frame.plot_traject(interaction)

