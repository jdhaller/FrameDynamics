"""
Date: 2022/12
Author: Jens D. Haller
Mail: jens.haller@kit.edu / jhaller@gmx.de
Institution: Karlsruhe Institute of Technology

"""

import numpy as np
from FrameDynamics.Frame import Frame
from FrameDynamics.Elements import Delay, Pulse, Shape

class Block(Frame):

    """
    The created objects can be used for alignment in the Frame class.

    Args for init:
        spins (list): specify all spins in a list (e.g. ["I", "J", "S])
          that are used in the given Block pulse sequence block.

    Returns:
        Block-Object: can be passed to "align" method of Frame class.
    """

    def __init__(self, Frame_object, spins_in_block):

        if not isinstance(spins_in_block, list):
            raise AssertionError("Please define a list of spins\
            for the Block class (e.g. [\"H1\"]).")

        if sum([True for spin in spins_in_block if not spin in Frame_object._Spins]):
            raise ValueError("Initiate Block class with spin(s) in Frame class.")

        self._Sequence = []
        self._Spins = Frame_object._Spins
        self._SpinsBlock = spins_in_block
        self._Offsets = Frame_object._Offsets
        self._PtsPerRot = Frame_object._PtsPerRot

        self._Zeeman = Frame_object._Zeeman
        self._H = Frame_object._H
        self._Interactions = Frame_object._Interactions

    # ====================================================================


    def set_interaction(self, spin1, spin2, iType="Jweak"):
        """
        Can not be set in Block class! Use Frame class instead.

        Raises:
            PermissionError: Use Frame class!
        """

        raise PermissionError("The Block class cannot set interactions.\
             Please, set interactions via a Frame object instead.")
    # ====================================================================


    def set_offset(self, spin1, Off):
        """
        Can not be set in Block class! Use Frame class instead.

        Raises:
            PermissionError: Use Frame class!
        """

        raise PermissionError("The Block class cannot set offsets.\
             Please, set interactions via a Frame object instead.")
    # ====================================================================


    def start(self, MP=True, CPUs=None, traject=False):
        """
        Can not be used in Block class! Use Frame class instead.

        Raises:
            PermissionError: Use Frame class!
        """

        raise PermissionError("The Block class cannot be started.\
             Please, pass the Block objects to Frame.align() instead.")
    # ====================================================================


    def delay(self, length):

        """
        Element for the creation of a pulse sequence.
        A delay can be defined with a duration / length (length).

        Args:
            length (float): the length of the delay given in seconds.

        """

        # create Delay-object
        element = Delay(length)
        element.setPTS(self._Offsets, self._SpinsBlock, self._PtsPerRot)
        element.aligned = self._SpinsBlock

        self._Sequence.append( element )
    # ====================================================================


    def pulse(self, spins: list, degree: float, amplitude: float,
             phase: float):

        """
        Element for the creation of a pulse sequence.
        A pulse can be defined with given rotation angle (degree),
        rf amplitude (amplitude) and pulse phase (phase).

        Args:
            spins (list): list containing the spins on which the pulse is
            applied (e.g. ["I", "J"])

            degree (float): specifies rotation angle of pulse.

            amplitude (float): specifies rf amplitude in Hz

            phase (float): specifies pulse phase (e.g. 0-> x, 1-> y, 2-> -x)

        Raises:
            ValueError: if given spins are not defined in Frame-object.
        """

        # ====================================================================
        # Verify that all spins are in self._SpinsBlock
        if set(spins) != set(self._SpinsBlock):
            print(" WARNING: Block-obj was initiated with different spins",\
            "than used py pulse-method.\n",\
            "Spins defined in the pulse() are ignored.")

        length = 1 / amplitude * degree / 360

        # create Pulse-object
        element = Pulse(set(self._SpinsBlock), length, amplitude, phase)
        element.setPTS(self._Offsets, self._SpinsBlock, self._PtsPerRot)
        element.aligned = self._SpinsBlock

        self._Sequence.append( element )
    # ====================================================================


    def shape(self, spins: list, shape: list, length:float, amplitude: float,
             phase: float):

        """
        Element for the creation of a pulse sequence.
        A shaped pulse (shape) can be defined with pulse length (length),
        rf amplitude (amplitude) and pulse phase (phase).

        Args:
            spins (list): list containing the spins on which the pulse is
              applied (e.g. ["I", "J"])

            shape (list): list containing all elements of the shaped pulse
              (e.g. [[amp, ph], ... [amp, ph]]). The method "load_shape" can
              be used to read in a suitable file in Bruker format.

            length (float): pulse length given in seconds.
              amplitude (float): specifies rf amplitude in Hz

            phase (float): specifies pulse phase (e.g. 0-> x, 1-> y, 2-> -x)

        Raises:
            ValueError: if given spins are not defined in Frame-object.
        """

        # ====================================================================
        # Verify that all spins are in self._SpinsBlock
        if set(spins) != set(self._SpinsBlock):
            print(" WARNING: Block-obj was initiated with different spins",\
            "than used py shape-method.\n",\
            "Spins defined in the shape() are ignored.")

        # create Shape-object
        element = Shape(set(self._SpinsBlock), shape, length, amplitude, phase)
        element.setPTS(self._Offsets, self._SpinsBlock, self._PtsPerRot)
        element.aligned = self._SpinsBlock

        self._Sequence.append( element )
    # ====================================================================



    def _returnDelay(self, length):
        # self._returnDelay() is called in self.align().
        # Difference to self.delay is that a value is returned and not
        # appended to self._Sequence

        # ====================================================================

        element = Delay(length)
        element.setPTS(self._Offsets, self._SpinsBlock, self._PtsPerRot)
        element.aligned = self._SpinsBlock

        return element
    # ====================================================================


