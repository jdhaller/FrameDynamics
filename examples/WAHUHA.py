from FrameDynamics.Frame import Frame

# initialize frame
frame = Frame(["I", "J"])

# setup interaction
interaction = frame.set_interaction("I", "J", "Dstrong")

# define pulse sequence and simulate
tau = 5 * 10**(-5)
frame.delay(tau)
frame.pulse(["I", "J"], 90, 10**(5), 0)
frame.delay(tau)
frame.pulse(["I", "J"], 90, 10**(5), 3)
frame.delay(2*tau)
frame.pulse(["I", "J"], 90, 10**(5), 1)
frame.delay(tau)
frame.pulse(["I", "J"], 90, 10**(5), 2)
frame.delay(tau)
frame.start(traject=True)

# plotting
frame.plot_traject(interaction, save="WAHUHA.png")
frame.plot_traject(interaction, operators=["xy","yx","xz","zx","yz","zy",])

