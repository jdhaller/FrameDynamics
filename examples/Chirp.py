# simulate homonuclear coupling evolution during chirp pulse

import numpy as np
from FrameDynamics.Frame import Frame

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

if __name__ == "__main__":    # required in Windows for multiprocessing
    frame.start(traject=True)

# plotting
frame.plot_H0_2D(interaction, save="Chirp_TF2D.png")
frame.plot_H0_1D(interaction, "J", offset=0., save="Chirp_TF1D.png")
frame.plot_traject(interaction)

