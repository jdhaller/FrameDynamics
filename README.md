# FrameDynamics

FrameDynamics is a python package that provides numerical simulations for the 
field of pulse sequence development in magnetic resonance. 

A coupling Hamiltonian is modulated in the toggling or interaction frame 
according to the specified pulse sequence and offset frequencies. 

The trajectory of the time-dependent Hamiltonian can be plotted or used 
to calculate the zeroth order average Hamiltonian (higher order terms might be 
available in following versions of FrameDynamics).

Theoretical background can be found in the publication (coming soon...).

## Installation

The python package can be installed via PyPI:

```
pip install FrameDynamics
```
## Simulations
Two examples shall be given: the WAHUHA sequence and a heteronuclear echo consisting of a shaped pulse and a hard 180° pulse. 

More examples can be found in the FrameDynamics github-repository ([link](https://github.com/jdhaller/FrameDynamics/tree/main/examples)).

### Example #1: WAHUHA sequence

Initialize frame:
```Python
from FrameDynamics.Frame import Frame
frame = Frame(["I", "J"]) 
```

Specify the interaction:
```Python
interaction = frame.set_interaction("I", "J", "Dstrong")
```

Define the pulse sequence:
```Python
tau = 5 * 10**(-5)
frame.delay(tau)
frame.pulse(["I", "J"], degree=90, amplitude=10**(5), phase=0)
frame.delay(tau)
frame.pulse(["I", "J"], degree=90, amplitude=10**(5), phase=3)
frame.delay(2*tau)
frame.pulse(["I", "J"], degree=90, amplitude=10**(5), phase=1)
frame.delay(tau)
frame.pulse(["I", "J"], degree=90, amplitude=10**(5), phase=2)
frame.delay(tau)
``` 

Start the simulations and plot trajectories without using multiprocessing (default: `MP=False`).
```Python
frame.start(traject=True)
frame.plot_traject(interaction, save="WAHUHA.png")
```

## Example #2: Reburp pulse

Load Frame and Block class. Block class is used to align different blocks
in the pulse sequence (e.g. Reburp pulse and 180° hard pulse in heteronuclear
echo)
```Python
import numpy as np
from FrameDynamics.Frame import Frame
from FrameDynamics.Block import Block
frame = Frame(["I", "S"])
```

Specify the interaction:
```Python
interaction = frame.set_interaction("I", "S", "Jweak")
```

Specify offset frequencies:
```Python
off = 5000
offsetsI = np.linspace(-off, off, 61)
offsetsS = np.linspace(-off, off, 61)
frame.set_offset("I", offsetsI)
frame.set_offset("S", offsetsS)
```

Load pulse shape to array:
```Python
Reburp = frame.load_shape("Reburp.1000")
```

**After** the interaction and offsets are set for the Frame object, one can now
initialize the Block class for each block. The frame-object has to be passed to the Block() class:

```Python
block1 = Block(frame, ["I"])
block2 = Block(frame, ["S"])
```

Define a Reburp pulse on `"I"` and hard pulse on `"S"` in first two lines.
Then center-align both block-elements (Reburp and hard pulse) within the frame-object.
```Python
block1.shape(["I"], Reburp, length=1000*10**(-6), amplitude=6264.8, phase=1)
block2.pulse(["S"], degree=180, amplitude=10000, phase=1)
frame.align(block1, block2, alignment="center")
```

Start the simulations using multiprocessing (`MP=True`). 
If using multiprocessing on Windows, the scope has to be resolved (`if __name__ == "__main__"`). Note, plotting and data retrieval has to be done in the same scope.

```Python
if __name__ == "__main__":, 
    frame.start(MP=True, traject=True)

    # Create offset-dependent 2D graph of the zeroth order average
    # Hamiltonian (H0) that is plotted against both offsets
    frame.plot_H0_2D(interaction, zlim=1)

    # Create offset-dependent 1D graph of H0 where offset of spin "S" 
    # is fixed to specified value (offset=0.)
    frame.plot_H0_1D(interaction, "S", offset=0.)

    # Plot trajectories for specified interaction and operators
    # (the given operators are default values)
    frame.plot_traject(interaction, operators=["x1","y1","z1","xx","yy","zz"])

    # Retrieve trajectories and the resulting average Hamiltonian.
    # Dictionaries are returned for specified offsets and operators.
    time, traject = frame.get_traject(interaction, offsets={"I": 0, "S": 300}, operators=["1z", "zz"])
    average_Hamiltonian = frame.get_results(interaction, operators=["zz"])
```
