"""
Date: 2022/12
Author: Jens D. Haller
Mail: jens.haller@kit.edu / jhaller@gmx.de
Institution: Karlsruhe Institute of Technology

"""



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
        self.PTS = {}
        self.aligned = None
        self.amp = 0

    def calcPTS(self, offsets: dict, _Spins: list, ptsPerRot: int) -> None:
        for spin in _Spins:
            temp = np.sqrt(np.array(offsets[spin])**2 + self.amp**2)
            temp *= (ptsPerRot * self.length)
            self.PTS[spin] = temp.astype("int") + 2

    @abstractmethod
    def setPTS(self, spins: list, offsets: dict, ptsPerRot: int):
        """ set self.PTS dict """
    # ====================================================================


class Delay(Elements):
    """ 
    Inherits from abs. base class (Elements) with abs. method: setPTS
    """

    def __init__(self, length: float):
        super().__init__("delay", length)
    
    def setPTS(self, offsets: dict, _Spins: list, ptsPerRot: int) -> None:
        self.amp = 0
        self.calcPTS(offsets, _Spins, ptsPerRot)
    # ====================================================================


class Pulse(Elements):
    """ 
    Inherits from abs. base class (Elements) with abs. method: setPTS
    """

    def __init__(self, spins: set, length: float, amp: float, \
                 phase: float) -> None:

        super().__init__("pulse", length)
        self.spins = spins
        self.amp = amp
        self.phase = phase
    
    def setPTS(self, offsets: dict, _Spins: list, ptsPerRot: int) -> None:
        self.calcPTS(offsets, _Spins, ptsPerRot)
    # ====================================================================


class Shape(Elements):
    """ 
    Inherits from abs. base class (Elements) with abs. method: setPTS
    """

    def __init__(self, spins: set, shape: list, length: float, \
                 amp: float, phase: float):
        super().__init__("shape", length)
        self.spins = spins
        self.shape = shape
        self.amp = amp
        self.phase = phase
    
    def setPTS(self, offsets: dict, _Spins: list, ptsPerRot: int) -> None:
        timestep = self.length / len(self.shape)
        for spin in _Spins:     # _Spins == all spins (even those w/o pulse)
            temp = np.sqrt(np.array(offsets[spin])**2 + self.amp**2)
            temp *= (ptsPerRot * timestep)
            temp = temp.astype("int") + 1
            self.PTS[spin] = temp * len(self.shape)
    # ====================================================================

