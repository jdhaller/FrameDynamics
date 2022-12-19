from abc import ABC, abstractmethod
import numpy as np
from typing import List


class Elements(ABC):
    """
    Base class for pulse sequence elements.
    """

    def __init__(self, name, length):
        self.name = name
        self.length = length
        self.PTS = None

    @abstractmethod
    def calcPTS(self, spins: List[str], offsets: dict, ptsPerHz: int):
        """ set _PTS dict """



class Delay(Elements):
    """ 
    Inherits from abs. base class (Elements) with abs. method: calcPTS
    """

    def __init__(self, length: float):
        super().__init__("delay", length)
    
    def calcPTS(self, offsets: dict, _Spins: List[str], ptsPerHz: int, \
                aligned = None) -> None:
        self.PTS = {"aligned": aligned}
        for spin in _Spins:
            temp = ptsPerHz * np.abs(offsets[spin]) * self.length
            self.PTS[spin] = temp.astype("int") + 2



class Pulse(Elements):
    """ 
    Inherits from abs. base class (Elements) with abs. method: calcPTS
    """

    def __init__(self, spins: set, length: float, amplitude: float, \
                 phase: float):
        super().__init__("pulse", length)
        self.spins = spins
        self.amp = amplitude
        self.phase = phase
    
    def calcPTS(self, offsets: dict, _Spins: List[str], ptsPerHz: int, \
                aligned = None) -> None:

        self.PTS = {"aligned": aligned}
        PtsOfPulse = int(ptsPerHz * self.amp * self.length) + 1
        for spin in _Spins:     # _Spins == all spins (even those w/o pulse)
            temp = ptsPerHz * np.abs(offsets[spin]) * self.length
            temp = temp.astype("int") + 1
            temp[ temp<PtsOfPulse ] = PtsOfPulse
            self.PTS[spin] = temp


class Shape(Elements):
    """ 
    Inherits from abs. base class (Elements) with abs. method: calcPTS
    """

    def __init__(self, spins: set, shape: List[float, float], length: float, \
                 amplitude: float, phase: float):
        super().__init__("shape", length)
        self.spins = spins
        self.shape = shape
        self.amp = amplitude
        self.phase = phase
    
    def calcPTS(self, offsets: dict, _Spins: List[str], ptsPerHz: int, \
                aligned = None) -> None:

        timestep = self.length / len(self.shape)
        self.PTS = {"aligned": aligned}
        PtsOfPulse = int(ptsPerHz * self.amp * timestep) + 1
        for spin in _Spins:     # _Spins == all spins (even those w/o pulse)
            temp = ptsPerHz * np.abs(offsets[spin]) * timestep
            temp = temp.astype("int") + 1
            temp[ temp<PtsOfPulse ] = PtsOfPulse
            self.PTS[spin] = temp




