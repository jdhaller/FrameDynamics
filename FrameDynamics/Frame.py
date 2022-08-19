"""
Date: 2022/08
Author: Jens D. Haller
Mail: jens.haller@kit.edu
Institution: Karlsruhe Institute of Technology

"""


import multiprocessing as mp
import time
import numpy as np
import scipy.linalg as scla

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import ImageGrid


# ====================================================================
# ====================================================================
class Frame():

    """
    Numerical simulation of a toggling or interaction frame for 
    arbitrary pulse sequences.

    Args for init:    
        spins (list): specify all spins in a list (e.g. ["I", "J", "S])
    
    Returns:
        Frame-Object: object that encapsulates all methods for the simulation.

    """

    # ====================================================================
    # 
    # PUBLIC METHODS
    #
    # ====================================================================


    def set_interaction(self, spin1, spin2, iType="Jweak"):

        """
        Set an interaction for subsequent simulations.
        All interactions need to be specified prior to the pulse sequence.

        Args:
            spin1 (str): First spin of interaction.

            spin2 (str): Second spin of interaction.

            iType (str): interaction type which can either be "Jweak", 
              "Jstrong", "Dweak", "Dstrong" (Default: "Jweak").

        Raises:
            RuntimeError: all interactions need to be specified prior to the
              pulse sequence.

        Returns:
            tuple: which specifies the given interaction. The tuple can later
              be passed to plotting-methods (plot_traject, plot_AHT1D, 
              plot_AHT2D)
        """

        # ====================================================================
        if self._Sequence != []:
            raise RuntimeError("Please, specify interactions \
                 prior to the pulse sequence!")
        self._Interactions.append( (spin1, spin2, iType) )
    
        return (spin1, spin2, iType)
    # ====================================================================


    def set_offset(self, spin1, Off):

        """
        Set offsets for subsequent simulations.
        All offsets need to be specified prior to the pulse sequence.

        Args:
            spin (str): spin for which the offsets are set.

            Off (list): list of offsets for given spin.

        Raises:
            RuntimeError: all offsets need to be specified prior to the
            pulse sequence.
        """

        # ====================================================================
        if self._Sequence != []:
            raise RuntimeError("Please, specify the offset \
                 prior to the pulse sequence!")
        self._Offsets[spin1] = Off
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
        # Verify that all spins are in self._Spins
        if sum([True for spin in spins if not spin in self._Spins]):
            raise ValueError("Initiate Frame with given spin(s).")

        length = 1 / amplitude * degree / 360

        # Set number of Points
        PTS = {"aligned": None}
        PtsOfPulse = int(self._PtsPerHz * amplitude * length) + 1
        for spin in self._Spins:
            temp = self._PtsPerHz * np.abs(self._Offsets[spin]) * length
            temp = temp.astype("int") + 1
            temp[ temp<PtsOfPulse ] = PtsOfPulse
            PTS[spin] = temp

        self._Sequence.append( ("pulse", set(spins), amplitude, \
                               phase, length, PTS) )
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
        # Verify that all spins are in self._Spins
        if sum([True for spin in spins if not spin in self._Spins]):
            raise ValueError("Initiate Frame with given spin(s).")

        # Set number of Points
        PTS = {"aligned": None}
        timestep = length / len(shape)
        PtsOfPulse = int(self._PtsPerHz * amplitude * timestep) + 1
        for spin in self._Spins:
            temp = self._PtsPerHz * np.abs(self._Offsets[spin]) * timestep
            temp = temp.astype("int") + 1
            temp[ temp<PtsOfPulse ] = PtsOfPulse
            PTS[spin] = temp * len(shape)

        self._Sequence.append( ("shape", set(spins), shape, amplitude, \
                               phase, length, PTS) )
    # ====================================================================


    def delay(self, length):

        """
        Element for the creation of a pulse sequence.
        A delay can be defined with a duration / length (length).

        Args:
            length (float): the length of the delay given in seconds.

        """
        # ====================================================================
        # Set number of Points
        PTS = {"aligned": None}
        for spin in self._Spins:
            temp = self._PtsPerHz * np.abs(self._Offsets[spin]) * length
            PTS[spin] = temp.astype("int") + 2

        self._Sequence.append( ("delay", length, PTS) )
    # ====================================================================

    
    def align(self, other1, other2, alignment="center"):
        """
        Method to align pulse sequence elements.
        "align" takes two objects of the Block class. Each object contains
        an individual sequence. The shorter sequence is extended by delays 
        to match the longer sequence. Both sequences are later placed in the 
        main-sequence (of Frame class).

        Args:
            other1 (object): first object created by the Block class 
              which contains first pulse sequence block for alignment.

            other2 (object): secong object created by the Block class
              which contains second pulse sequence block for alignment.

            alignment (str): define the alignment. Can be either "center", 
              "right" or "left". Defaults to "center".
        """

        t1, t2 = self._checkObjs(other1, other2)
        D = t2 - t1

        # ====================================================================
        # Extend the shorter sequence by delays according to the alignment
        # argument ("center", "left", "right")
        if t1 < t2: length = t2
        else: length = t1

        if D != 0:
            if alignment == "left":
                if t1 < t2: other1._Sequence.append(other1._returnDelay(D))
                else: other2._Sequence.append(other2._returnDelay(-D))

            elif alignment == "right":
                if t1 < t2: other1._Sequence.insert(0, other1._returnDelay(D))
                else: other2._Sequence.insert(0, other2._returnDelay(-D) )

            elif alignment == "center":
                if t1 < t2: 
                    other1._Sequence.insert(0, other1._returnDelay(D/2) )
                    other1._Sequence.append( other1._returnDelay(D/2) )
                else:
                    other2._Sequence.insert(0, other2._returnDelay(-D/2) )
                    other2._Sequence.append( other2._returnDelay(-D/2) )
        # ====================================================================

        for seq in other1._Sequence: self._Sequence.append( seq )
        for seq in other2._Sequence: self._Sequence.append( seq )
    # ====================================================================


    def load_shape(self, path_to_file, separator=",", start="##XYPOINTS=", end="##END"):
        """
        Load a pulse shape from file in Bruker format.

        Args:
            path_to_file (str): specifiy path to file (including filename).

            separator (str, optional): character that is used to separate 
              amplitude and phase in file of shaped pulse. Defaults to ",".

            start (str, optional): specifies the line after which the shaped
              pulse starts in file of shaped pulse. Defaults to "##XYPOINTS=".

            end (str, optional): specifies the line where the shaped pulse 
              ends in file of shaped pulse. Defaults to "##END".

        Returns:
            shape (list): list of shaped pulse ([ [amp, ph], ..., [amp, ph] ])
        """

        data = []; 

        with open(path_to_file, 'r') as tmp:
            load_data = tmp.read()
        load_data = load_data.split('\n')
        
        flag = False
        for line in load_data:
            if end in line: flag = False
            if flag:
                line = line.split(separator)
                try: line = [float(slic) for slic in line]
                except: line = [slic.replace(' ', '') for slic in line]
                data.append(line)
            if start in line: flag = True

        return data
    # ====================================================================


    def start(self, MP=True, CPUs=None, Traject=False):

        """
        Start simulations after the pulse sequence has been defined.

        Args:
            MP (bool, optional): use multiprocessing. Defaults to True.

            CPUs (int, optional): define number of CPUs for multiprocessing.
              Defaults to None and all CPUs except one is used.

            Traject (bool, optional): has to be set to True if the 
              trajectory of the Hamiltonian in the interaction frame is 
              needed. Defaults to False.
        
        Raises:
            AssertionError: an interaction or pulse sequence needs to be
              defined before the simulation can be started.
        """

        self._MP = MP
        self._flagT = Traject
        self._START = time.time()

        if self._Interactions == [] or self._Sequence == []:
            raise AssertionError( "Define interaction or sequence first!" )

        print("\n  === >>>    Starting simulations    <<< ===  \n")

        # ====================================================================
        # Set the number of CPUs according to input or to "CPUs_available - 1"
        if CPUs == None:
            if mp.cpu_count() == 1: self.CPUs = 1
            else: self._CPUs = mp.cpu_count() - 1

        else:
            MP = True
            self._CPUs = CPUs
        
        if MP:
            self._Out = mp.Manager().dict()
            pool = mp.Pool(processes=self._CPUs)

            self._Results = mp.Manager().dict()
            pool2 = mp.Pool(processes=self._CPUs)
            if self._flagT: self._Traject = mp.Manager().dict()

        else:
            self._Out = {}
            self._Results = {}
            if self._flagT: self._Traject = {}
        # ====================================================================


        # ====================================================================
        # Calculate the trajectories of all spins and offsets individually
        # Typical structure: self._Offsets = {spin: offsets}
        # e.g. self._Offsets = {'H1': [0], 'H2': [-2, -1, 0, 1, 2]}

        for spin, offsets in self._Offsets.items():
            for o, offset in enumerate(offsets):
                
                # Initialize basis and hamilton operators, 
                # as well as trajectory
                pts = self._calcPoints(spin, o)
                B, traject = self._zeroTraject(pts)
                args = ((spin, o), B, traject, offset,)

                # Start simulations -> output: self._Out
                if MP:
                    # Using a pool for multiprocessing
                    pool.apply_async(self._simulate, args=args)

                else: 
                    # No multiprocessing
                    self._simulate(*args)

        if MP:
                pool.close()    # Close pool of workers
                pool.join()     # Join pool
                # self._Out.clear()

        # ====================================================================

        self._STEP = time.time()
        diff = self._STEP - self._START
        print("   Single-spin trajectory: {:.3f} seconds ".format(diff))

        # ====================================================================        
        # create zeroth order average Hamiltonian
        
        for interaction in self._Interactions:
            spin1, spin2, iType = interaction

            for o1 in range(len(self._Offsets[spin1])):
                for o2 in range(len(self._Offsets[spin2])):
                    args = (interaction, o1, o2)

                    # Start calculation of average Hamiltonian
                    if MP:
                        pool2.apply_async(self._expandBasis, args=args)
                    else:
                        self._expandBasis(*args)
            
        if MP:
            pool2.close()    # Close pool of workers
            pool2.join()     # Join pool
        # ====================================================================

        self.END = time.time()
        diff = self.END - self._STEP
        print("   Basis Expansion: {:.3f} seconds ".format(diff))

        diff = self.END - self._START
        print("\n  === >>>   Total: {:.3f} seconds   <<< ===\n".format(diff))
    # ====================================================================


    def get_results(self, returnType="all"):

        """
        Outputs average Hamiltonians as a dictionary or numpy array (3D).

        Args:
            returnType (tuple): can be used to specify the output. If an 
              an interaction (tuple) is passed, the respective Defaults to "all".
        
        Returns:
            numpy arrad (3D) if an interaction is specified in returnType,
            else a dictionary containing all interactions. The 3 dimensions
            in the numpy array are given as (9, len(offsets), len(offsets))
            where the offsets were defined by "set_offsets" for respective
            spins in interaction. In the first dimension the 9 bilinear 
            basis operators are stored:
            "xx": [0], "xy": [1], "xz": [2],
            "yx": [3], "yy": [4], "yz": [5],
            "zx": [6], "zy": [7], "zz": [8].
        """

        if returnType == "all": Iterable = self._Interactions
        else: Iterable = [returnType]

        # ====================================================================
        Results = {}

        for interaction in Iterable:
            spin1, spin2, iType = interaction
            
            LO1 = len(self._Offsets[spin1])
            LO2 = len(self._Offsets[spin2])
            Results[interaction] = np.zeros([9, LO1, LO2])

            for o1 in range(LO1):
                for o2 in range(LO2):

                    Results[interaction][:, o1, o2] = \
                        self._Results.get((interaction, o1, o2))

        if returnType == "all":
            return Results
        else: 
            return Results[interaction]
    # ====================================================================
    

    def get_traject(self):
        
        if self._flagT == False:
            raise AssertionError("Trajectories were not calculated. \
                Use the option in Frame.start(Traject=True).")
        else: 
            return dict(self._Traject)
    # ====================================================================


    def plot_traject(self, interaction, \
                    operators=["x1","y1","z1","xx","yy","zz"], offsets=None,\
                    save=None, show=True):
        """
        Plot trajectory of Hamiltonian in the toggling / interaction frame.

        Args:
            interaction (tuple): specify the interaction for which the 
              trajectory is supposed to be plotted.

            operators (list): specify a list of all operators that are 
              supposed to be plotted. Valid input is:
              ["x1","y1","z1","1x","1y","1z",
               "xx","yy","zz","xy","xy","xz", "zx", "yz", "zy"]
              Defaults to ["x1","y1","z1","xx","yy","zz"].

            offsets (dict): is a dictionary in which for each spin an offset
              can be defined for which the trajectory is supposed to be 
              plotted (e.g. offsets = {spin1: 0, spin2: 100}). 
              Defaults to None (which implies {spin1: 0, spin2: 0}).

            save (str, optional): define a filename to save figure. 
              Defaults to None.

            show (bool, optional): show figure. Defaults to True.

        Raises:
            ValueError: if specified operators are not valid.
        """

        # offsets = {spin1: 0, spin2: 100}

        X, Y, labels, AHT = {}, {}, {}, {}
        n = len(operators)
        spin1, spin2 = interaction[0], interaction[1]

        # Index of respective operators in self._Out (linear operators)
        # or in self._Traject (bilinear operators).
        index = {            "1x": (0,), "1y": (1,), "1z": (2,), 
                 "x1": (0,), "xx": (0,), "xy": (1,), "xz": (2,), 
                 "y1": (1,), "yx": (3,), "yy": (4,), "yz": (5,), 
                 "z1": (2,), "zx": (6,), "zy": (7,), "zz": (8,),}
        # ====================================================================

        # Check for invalid input in operators
        valid = set(index.keys())
        if not valid.issuperset(set(operators)):
            raise ValueError("The entered operators are not valid.\n\
            Please use some of the following: \n {}".format(valid))
        # ====================================================================

        # Set offsets to 0 (default) or as specified in offsets-input
        if offsets == None:
            idx1 = self._find_nearest(self._Offsets[spin1], 0)
            idx2 = self._find_nearest(self._Offsets[spin2], 0)
        elif spin1 in offsets.keys() and spin2 in offsets.keys():
            idx1 = self._find_nearest(self._Offsets[spin1], offsets[spin1])
            idx2 = self._find_nearest(self._Offsets[spin2], offsets[spin2])
        elif spin1 in offsets.keys():
            idx1 = self._find_nearest(self._Offsets[spin1], offsets[spin1])
            idx2 = self._find_nearest(self._Offsets[spin2], 0)
        elif spin2 in offsets.keys():
            idx1 = self._find_nearest(self._Offsets[spin1], 0)
            idx2 = self._find_nearest(self._Offsets[spin2], offsets[spin2])
        # ====================================================================

        # Retrieve data from self._Out (linear operators) and from
        # self._Traject (bilinear operators) for plotting
        for i in range(n):
            if operators[i][0] == "1":
                X[i] = self._Out.get((spin2, idx2))[-1]
                X[i] *= 1000
                Y[i] = self._Out.get((spin2, idx2))[index[operators[i]]]
                labels[i] = "${}_{}$".format(spin2, operators[i][1])
            elif operators[i][1] == "1":
                X[i] = self._Out.get((spin1, idx1))[-1]
                X[i] *= 1000
                Y[i] = self._Out.get((spin1, idx1))[index[operators[i]]]
                labels[i] = "${}_{}$".format(spin1, operators[i][0])
            else:
                X[i] = self._Traject.get(\
                    (interaction, idx1, idx2)\
                                        )[-1]
                X[i] *= 1000
                Y[i] = self._Traject.get(\
                    (interaction, idx1, idx2)\
                                        )[index[operators[i]]]
                labels[i] = "$2{}_{}{}_{}$".format(spin1, operators[i][0],\
                                                   spin2, operators[i][1])
        # ====================================================================

        # Calculate average value for given operator
        for i in range(n):

            T = X[i][-1]
            dT = self._diffT(X[i])
            iH = self._interH1(Y[i])
            AHT[i] = round(self._integrate1(dT, iH) / T * 100, 1)
        # ====================================================================

        # Start plotting the graphs
        fig = plt.figure(figsize=(10, 0.5+n*1.5))
        ax = ImageGrid(fig, 111,          # as in plt.subplot(111)
                nrows_ncols=(n,1),
                axes_pad=0.23,
                share_all=True,
                aspect=False)

        for i in range(n): 
            ax[i].plot(X[i], Y[i], color='#80000f', lw=2.)
            ax[i].fill_between(X[i], Y[i], color='#cc000f')
            ax[i].set_ylabel("k(t)", rotation=0, size=14)
            ax[i].yaxis.set_label_coords(-0.05,0.85)
            ax[i].plot([X[i][0],X[i][-1]],[1,1],"k--",lw=0.5)
            ax[i].plot([X[i][0],X[i][-1]],[0,0],"k",lw=1)
            ax[i].plot([X[i][0],X[i][-1]],[-1,-1],"k--",lw=0.5)
            ax[i].text(-0.1,0.5,labels[i],size=18,transform=ax[i].transAxes,\
                       ha="center")
            L = "({}%)".format(AHT[i])
            ax[i].text(-0.1,0.31, L, size=12,transform=ax[i].transAxes, \
                       ha="center")
        
        ax[n-1].set_xlabel("time / ms", fontsize=15)
        ax[0].set_xlim([X[i][0], X[i][-1]])
        
        plt.tight_layout()

        if save != None: plt.savefig(save, dpi=300)
        if show: plt.show()
    # ====================================================================


    def plot_AHT1D(self, interaction, fixedSpin, offset=0, \
                   save=None, show=True, **kwargs):
        """
        Plot the average Hamiltonian for specified interaction against
        the offset of one spin (1D).

        Args:
            interaction (tuple): specify the interaction for which the 
              trajectory is supposed to be plotted.

            fixedSpin (str): define the name of the spin in the interaction,
              whose offset is fixed. The offset of the other spin is used as
              x-axis.

            offset (int, optional): define the offset value for the spin with
              constant offset. Defaults to 0.

            show (bool, optional): show figure. Defaults to True.

        Raises:
            ValueError: if defined spin is not in interaction.
        """

        # Get index and retrieve data
        idx = self._find_nearest(self._Offsets[fixedSpin], offset)
        temp_all = self.get_results(interaction)

        if interaction[0] == fixedSpin: 
            spinX = interaction[1]
            temp = temp_all[:, idx, :]
        elif interaction[1] == fixedSpin: 
            spinX = interaction[0]
            temp = temp_all[:, :, idx]
        else: 
            raise ValueError("Selected spin is not part of the interaction.")
        
        X = self._Offsets[spinX] / 1000
        # ====================================================================
        

        # Start plotting the graphs
        fig = plt.figure(figsize=(10, 8))
        ax = ImageGrid(fig, 111,          # as in plt.subplot(111)
                nrows_ncols=(3,3),
                axes_pad=0.23,
                share_all=True,
                aspect=False)

        labels = [r"$2{}_x{}_x$".format(fixedSpin, spinX), 
                  r"$2{}_x{}_y$".format(fixedSpin, spinX), 
                  r"$2{}_x{}_z$".format(fixedSpin, spinX), 
                  r"$2{}_y{}_x$".format(fixedSpin, spinX), 
                  r"$2{}_y{}_y$".format(fixedSpin, spinX), 
                  r"$2{}_y{}_z$".format(fixedSpin, spinX), 
                  r"$2{}_z{}_x$".format(fixedSpin, spinX), 
                  r"$2{}_z{}_y$".format(fixedSpin, spinX), 
                  r"$2{}_z{}_z$".format(fixedSpin, spinX), 
                 ]

        for i in range(9):
            ax[i].plot([X[0], X[-1]], [0, 0], "k--", lw=1)
            ax[i].plot(X, temp[i], lw=2.5, color="#1425a4", **kwargs)
            ax[i].text(0.05, 0.05, labels[i], transform=ax[i].transAxes, \
                       size = 20)
        
        ax[0].set_xlim([X[0], X[-1]])
        ax[7].set_xlabel("offset (%s) / kHz" % spinX, size=15)
        ax[6].set_xlabel("offset (%s) / kHz" % spinX, size=15)
        ax[8].set_xlabel("offset (%s) / kHz" % spinX, size=15)

        ax[0].set_ylabel("$k_0$ / a. u.", size=15)
        ax[3].set_ylabel("$k_0$ / a. u.", size=15)
        ax[6].set_ylabel("$k_0$ / a. u.", size=15)

        plt.tight_layout()

        if save != None: plt.savefig(save, dpi=300)
        if show: plt.show()
    # ====================================================================


    def plot_AHT2D(self, interaction, levels=21, zlim=None, save=None, \
                   show=True):
        """    
        Plot the average Hamiltonian for specified interaction against
        both offsets of spin1 and spin2 (2D).

        Args:
            interaction (tuple): specify the interaction for which the 
              trajectory is supposed to be plotted.

            levels (int, optional): contour levels. Defaults to 21.

            zlim (float, optional): limit for z-axis. 
            Defaults to None (choose z limit automatically).

            show (bool, optional): show figure. Defaults to True.

        Raises:
            ValueError: if defined spin is not in interaction.

            Args:
            interaction (_type_): _description_
            levels (int, optional): _description_. Defaults to 21.
            zlims (_type_, optional): _description_. Defaults to None.
            save (_type_, optional): _description_. Defaults to None.
            show (bool, optional): _description_. Defaults to True.

        """

        # Create colormap and retrieve data for plotting
        RWB = self._getColorMap(levels-1)

        spinY, spinX = interaction[0], interaction[1]
        temp = self.get_results(interaction)

        X, Y = self._Offsets[spinX] / 1000, self._Offsets[spinY] / 1000

        if zlim != None: vmax = zlim
        else: vmax = np.max(np.abs(temp))
        vals = np.linspace(-vmax, vmax, levels)
        # ====================================================================

        # Start plotting the graphs
        fig = plt.figure(figsize=(10, 9))

        ax = ImageGrid(fig, 111,          # as in plt.subplot(111)
                nrows_ncols=(3,3),
                axes_pad=0.23,
                share_all=True,
                cbar_location="right",
                cbar_mode="edge",
                cbar_size="7%",
                cbar_pad=0.2,
                aspect=False)

        labels = [r"$2{}_x{}_x$".format(spinY, spinX), 
                  r"$2{}_x{}_y$".format(spinY, spinX), 
                  r"$2{}_x{}_z$".format(spinY, spinX), 
                  r"$2{}_y{}_x$".format(spinY, spinX), 
                  r"$2{}_y{}_y$".format(spinY, spinX), 
                  r"$2{}_y{}_z$".format(spinY, spinX), 
                  r"$2{}_z{}_x$".format(spinY, spinX), 
                  r"$2{}_z{}_y$".format(spinY, spinX), 
                  r"$2{}_z{}_z$".format(spinY, spinX), 
                 ]

        for i in range(9):
            cb = ax[i].contourf(X, Y, temp[i], levels=vals, cmap=RWB)
            ax[i].text(0.05, 0.85, labels[i], transform=ax[i].transAxes, \
                       size = 20)
        
        ax[7].set_xlabel("offset (%s) / kHz" % spinX, size=15)
        ax[6].set_xlabel("offset (%s) / kHz" % spinX, size=15)
        ax[8].set_xlabel("offset (%s) / kHz" % spinX, size=15)

        ax[0].set_ylabel("offset (%s) / kHz" % spinY, size=15)
        ax[3].set_ylabel("offset (%s) / kHz" % spinY, size=15)
        ax[6].set_ylabel("offset (%s) / kHz" % spinY, size=15)
        
        cbar1 = ax[2].cax.colorbar(cb)
        cbar2 = ax[5].cax.colorbar(cb)
        cbar3 = ax[8].cax.colorbar(cb)

        cbar1.ax.set_yticks(np.linspace(-vmax, vmax, 11))
        cbar2.ax.set_yticks(np.linspace(-vmax, vmax, 11))
        cbar3.ax.set_yticks(np.linspace(-vmax, vmax, 11))

        plt.tight_layout()

        if save != None: plt.savefig(save, dpi=300)
        if show: plt.show()
    # ====================================================================




    # ====================================================================
    # 
    # PRIVATE METHODS
    #
    # ====================================================================


    # Pauli matrices
    mIx = 0.5 * np.array([[0,1.],[1.,0]], dtype="complex64")
    mIy = 0.5 * 1j * np.array([[0,-1.],[1.,0]], dtype="complex64")
    mIz = 0.5 * np.array([[1.,0],[0,-1.]], dtype="complex64")
    mIp = np.array([[0,1],[0,0]], dtype="complex64")
    mIm = np.array([[0,0],[1,0]], dtype="complex64")
    mIa = np.array([[1,0],[0,0]], dtype="complex64")
    mIb = np.array([[0,0],[0,1]], dtype="complex64")
    

    def __init__(self, spins):
        """
        spins = ["I", "J"]
        """
        # ====================================================================
        self._Spins = spins         # List for all spins
        self._Offsets = {spin: [0] for spin in spins}  # Offsets for each spin
        self._Sequence = []         # List for pulse sequence

        self._Zeeman = np.zeros((2, 2), dtype="complex64")  # alloc
        self._H = (self.mIz, self.mIy, self.mIx)    # constant single-spin Ham
        self._Interactions = []     # List of considered interactions
        self._PtsPerHz = 21         # How many points per oscillation
        self._flagT = False         # Flag for trajectories
    # ====================================================================


    def _setZeeman(self, offset):
        return 2*np.pi * offset * self.mIz


    def _zeroTraject(self, Pts):

        # Returns Cartesian basis operators for a single spin
        # and empty trajectory lists.
        B = np.stack([self.mIx, self.mIy, self.mIz])    # is propagated
        traject = np.zeros((10, Pts))

        return B, traject

     
    def _pulseHam(self, phase, amplitude):
        return 2*np.pi* (
                         np.cos(2*np.pi* phase/4.) * self.mIx + \
                         np.sin(2*np.pi* phase/4.) * self.mIy \
                        ) * amplitude


    @staticmethod
    def _scalarProduct(h, b):
        # Scalar Product of Hamilton Operator (h) on Basis (b)
        scalar = np.trace(b.T.conj() @ h) / np.trace(b.T.conj() @ b)
        return scalar.real


    def _measure(self, B, traject, p):	
        for i, h in enumerate(self._H): 
            traject[3*i+0, p] = self._scalarProduct( h, B[0] )
            traject[3*i+1, p] = self._scalarProduct( h, B[1] )
            traject[3*i+2, p] = self._scalarProduct( h, B[2] )
        return traject


    @staticmethod
    def _propagate(B, U):
        for i, b in enumerate(B): B[i] = U @ b @ U.T.conj()
        return B
    

    def _diffT(self, t):
        # turn timeseries to timesteps
        return t[1:] - t[:-1]

    @staticmethod
    def _interH(A): 
        # interpolate time-dependent Hamiltonian (-> length is reduced by 1)
        return (A[:, 1:] + A[:, :-1]) / 2

    @staticmethod
    def _interH1(A): 
        # interpolate time-dependent Hamiltonian (-> length is reduced by 1)
        return (A[1:] + A[:-1]) / 2	

    @staticmethod
    def _integrate(dT, iH):
        A = np.zeros(9)
        for i, h in enumerate(iH): 
            A[i] = np.sum( np.multiply(h, dT) ) 
        return A

    @staticmethod
    def _integrate1(dT, iH):
        return np.sum( np.multiply(iH, dT) ) 
    
    @staticmethod
    def _interpolate(t, A):
        out = np.zeros((10, len(t)))

        for i in range(9): 
            out[i] = np.interp(t, A[-1], A[i])
        out[-1] = t
        return out


    def _calcPoints(self, spin, o):
        pts = 1
        for (*_, PTS) in self._Sequence:
            if PTS["aligned"] == None or spin in PTS["aligned"]:
                pts += PTS[spin][o]
        return pts


    def _Pulse(self, offset, amplitude, phase, length, pts):
        # Frame._Pulse() is called by Frame._simulate()

        # ====================================================================
        # Set Zeeman Hamiltonian according to offset
        Zeeman = self._setZeeman(offset)
        timestep = length / int(pts)
        U = scla.expm(-1j * timestep * (self._pulseHam(phase, amplitude) + Zeeman) )
        
        return U, timestep, pts
    # ====================================================================


    def _Delay(self, offset, length, pts):
        # Frame._Delay() is called by Frame._simulate()

        # ====================================================================
        # Set Zeeman Hamiltonian according to offset
        Zeeman = self._setZeeman(offset)
        timestep = length / int(pts)
        U = scla.expm(-1j * timestep * Zeeman )
        
        return U, timestep, pts
    # ====================================================================


    def _Transform(self, U, B, traject, p, timestep, pts):
        # Frame._Transform() is called by Frame._simulate()

        # ====================================================================
        for i in range(pts):

            # Propagate basis operators
            B = self._propagate(B, U)

            traject = self._measure(B, traject, p)
            traject[-1, p] = traject[-1, p-1] + timestep
            p += 1
        
        return B, traject, p
    # ====================================================================


    def _Shape(self, B, traject, p, \
               offset, shape, amplitude, phase, length, pts):
        # Frame._Shape() is called by Frame._simulate()

        # ====================================================================
        # Set Zeeman Hamiltonian according to offset
        Zeeman = self._setZeeman(offset)
        timestep = length / pts
        N = int(pts / len(shape))
        if N != 1: print("N != 1  -> ", N )
        # ====================================================================

        # return U, timestep, pts

        # ====================================================================
        for pul in shape:
            U = scla.expm(-1j * timestep * \
                          (self._pulseHam(pul[1]/90 + phase, \
                                          pul[0]/100 * amplitude) + Zeeman) )
            for i in range(N):

                # Propagate basis operators
                B = self._propagate(B, U)

                traject = self._measure(B, traject, p)
                traject[-1, p] = traject[-1, p-1] + timestep
                p += 1
        
        return B, traject, p
    # ====================================================================


    def _simulate(self, index, B, traject, offset):
        # Frame._simulate() is called by Frame.start()
        # with and without multiprocessing

        # Set first point
        p = 0
        traject = self._measure(B, traject, p)
        p += 1

        # ====================================================================
        # Simulate Pulse Sequence in single-spin basis
        spin = index[0]

        for (action, *args) in self._Sequence:
            aligned = args[-1]["aligned"]
            if aligned == None or spin in aligned:
                pts = args[-1][spin][index[1]]

            # Check if pulse is applied on current spin (index[0])
            # "pulse": args = (set(spins), amplitude, phase, length, PTS)
            if action == "pulse" and spin in args[0]:
                U, timestep, pts = self._Pulse(offset, *(args[1:-1]), pts)
                B, traject, p = self._Transform(U, B, traject, p, \
                                                   timestep, pts)
            
            # If pulse is not applied on current spin -> use a delay
            # unless alignment is used!
            # args[-2] = length
            elif action == "pulse" and aligned == None:
                U, timestep, pts = self._Delay(offset, args[-2], pts)
                B, traject, p = self._Transform(U, B, traject, p, \
                                                   timestep, pts)

            # Simulate a delay unless alignment is used!
            # "delay": args = (length, PTS)
            if action == "delay" and aligned == None:
                U, timestep, pts = self._Delay(offset, args[0], pts)
                B, traject, p = self._Transform(U, B, traject, p, \
                                                   timestep, pts)

            # Using alignment (align = True)!
            # Simulate an individual delay for the specified spin! 
            elif action == "delay" and spin in aligned:
                U, timestep, pts = self._Delay(offset, args[0], pts)
                B, traject, p = self._Transform(U, B, traject, p, \
                                                   timestep, pts)

            # Simulate a shaped pulse
            # "shape": args = (set(spins), shape, amplitude, phase, length, PTS)                                                   
            if action == "shape" and spin in args[0]:
                B, traject, p = self._Shape(B, traject, p, \
                                               offset, *(args[1:-1]), pts)

            # If the spin is not in spins (args[0]) then simulate 
            # a delay instead unless the alignment statement is used!
            elif action == "shape" and aligned == None:
                U, timestep, pts = self._Delay(offset, args[-2], pts)
                B, traject, p = self._Transform(U, B, traject, p, \
                                                   timestep, pts)         
            # ====================================================================

        if p != len(traject[-1]):
            print(" DIMENSION MIS-MATCH! ", p, len(traject[-1]))
        
        self._Out[index] = traject
    # ====================================================================


    def _expandBasis(self, interaction, o1, o2):
        # Frame._expandBasis() is called by Frame.start() 
        # with and without multiprocessing

        # Interaction: (spin1, spin2, interactionType)
        spin1, spin2, iType = interaction
        
        # ====================================================================
        # Different timepoints and number of timepoints are calculated in 
        # the two trajectories. This is due to the fact that less points 
        # need to be calculated for slow osciallations, i.e. small offsets.
        # Interpolation is, hence, required before multiplication.
        T1 = self._Out[(spin1, o1)]
        T2 = self._Out[(spin2, o2)]
        if round(T1[9][-1], 12) != round(T2[9][-1], 12): 
            print("WARNING: TIME-MISMATCH?")

        timeseries = np.sort(np.concatenate( [T1[9], T2[9]] ) )

        # Interpolation
        T1 = self._interpolate( timeseries, T1 )
        T2 = self._interpolate( timeseries, T2 )
        # ====================================================================

        # ====================================================================
        # Prepare output
        # Create time-dependent Hamiltonian in interaction frame (upper 
        # case X,Y,Z) based on a (time-independent) coupling Hamiltonian 
        # in rotating frame (lower case x,y,z).
        out = np.zeros((10, len(timeseries)))
        out[-1] = timeseries

        X1z = T1[0];  X1y = T1[3];  X1x = T1[6]
        Y1z = T1[1];  Y1y = T1[4];  Y1x = T1[7]
        Z1z = T1[2];  Z1y = T1[5];  Z1x = T1[8]

        X2z = T2[0];  X2y = T2[3];  X2x = T2[6]
        Y2z = T2[1];  Y2y = T2[4];  Y2x = T2[7]
        Z2z = T2[2];  Z2y = T2[5];  Z2x = T2[8]
        # ====================================================================

        # ====================================================================
        # scalar weak coupling
        if iType == "Jweak":
            out[0] = X1z*X2z   # XX
            out[1] = X1z*Y2z   # XY
            out[2] = X1z*Z2z   # XZ
            
            out[3] = Y1z*X2z   # YX
            out[4] = Y1z*Y2z   # YY
            out[5] = Y1z*Z2z   # YZ
            
            out[6] = Z1z*X2z   # ZX
            out[7] = Z1z*Y2z   # ZY
            out[8] = Z1z*Z2z   # ZZ

        # scalar strong coupling
        elif iType == "Jstrong":
            out[0] = X1z*X2z + X1y*X2y + X1x*X2x   # XX  
            out[1] = X1z*Y2z + X1y*Y2y + X1x*Y2x   # XY
            out[2] = X1z*Z2z + X1y*Z2y + X1x*Z2x   # XZ
                        
            out[3] = Y1z*X2z + Y1y*X2y + Y1x*X2x   # YX
            out[4] = Y1z*Y2z + Y1y*Y2y + Y1x*Y2x   # YY
            out[5] = Y1z*Z2z + Y1y*Z2y + Y1x*Z2x   # YZ
                        
            out[6] = Z1z*X2z + Z1y*X2y + Z1x*X2x   # ZX
            out[7] = Z1z*Y2z + Z1y*Y2y + Z1x*Y2x   # ZY
            out[8] = Z1z*Z2z + Z1y*Z2y + Z1x*Z2x   # ZZ

        # dipolar weak coupling
        elif iType == "Dweak":
            out[0] = 2*X1z*X2z   # XX
            out[1] = 2*X1z*Y2z   # XY
            out[2] = 2*X1z*Z2z   # XZ
            
            out[3] = 2*Y1z*X2z   # YX
            out[4] = 2*Y1z*Y2z   # YY
            out[5] = 2*Y1z*Z2z   # YZ
            
            out[6] = 2*Z1z*X2z   # ZX
            out[7] = 2*Z1z*Y2z   # ZY
            out[8] = 2*Z1z*Z2z   # ZZ

        # dipolar strong coupling
        elif iType == "Dstrong":
            out[0] = 2*X1z*X2z - X1y*X2y - X1x*X2x   # XX  
            out[1] = 2*X1z*Y2z - X1y*Y2y - X1x*Y2x   # XY
            out[2] = 2*X1z*Z2z - X1y*Z2y - X1x*Z2x   # XZ
                        
            out[3] = 2*Y1z*X2z - Y1y*X2y - Y1x*X2x   # YX
            out[4] = 2*Y1z*Y2z - Y1y*Y2y - Y1x*Y2x   # YY
            out[5] = 2*Y1z*Z2z - Y1y*Z2y - Y1x*Z2x   # YZ
                        
            out[6] = 2*Z1z*X2z - Z1y*X2y - Z1x*X2x   # ZX
            out[7] = 2*Z1z*Y2z - Z1y*Y2y - Z1x*Y2x   # ZY
            out[8] = 2*Z1z*Z2z - Z1y*Z2y - Z1x*Z2x   # ZZ

        if self._flagT: self._Traject[interaction, o1, o2] = out
        # ====================================================================

        # ====================================================================
        # Calculation of zeroth order average Hamiltonian 
        T = timeseries[-1]
        dT = self._diffT(timeseries)
        iH = self._interH(out[:9])

        self._Results[interaction, o1, o2] = self._integrate(dT, iH) / T
    # ====================================================================


    def _checkObjs(self, other1, other2):
        # ====================================================================
        # Blockd elements must be specified for different spins!
        # If not, raise Permission Error!
        temp1, temp2 = set(), set()
        time1, time2 = 0, 0

        for (action, *args) in other1._Sequence:
            time1 += args[-2]

            if action == "pulse" or action == "shape":
                temp1.update(args[0])

        for (action, *args) in other2._Sequence:
            time2 += args[-2]

            if action == "pulse" or action == "shape":
                temp2.update(args[0])

        if not temp1.isdisjoint(temp2):
            raise PermissionError("Cannot be aligned: \
            custom sequences must be specified for different spins!")
        
        return time1, time2
    # ====================================================================


    def _returnDelay(self, length):
        # self._returnDelay() is called in self.align().
        # Difference to self.delay is that a value is returned and not 
        # appended to self._Sequence

        # ====================================================================
        # Set number of Points
        PTS = {"aligned": self._SpinsBlock}
        for spin in self._SpinsBlock:
            temp = self._PtsPerHz * np.abs(self._Offsets[spin]) * length
            PTS[spin] = temp.astype("int") + 2

        return ("delay", length, PTS) 
    # ====================================================================
    

    @staticmethod
    def _interpColors(V, Z, f):
        return (V[0]+f*(Z[0]-V[0]), V[1]+f*(Z[1]-V[1]), V[2]+f*(Z[2]-V[2]) )
    # ====================================================================


    def _getColorMap(self, N):
        
        high = (20/255, 50/255, 180/255)
        zero = (1, 1, 1)
        low =  (160/255, 0, 0)

        highs = [self._interpColors(high,zero,f) for f in np.linspace(0,1,N)]
        lows = [self._interpColors(low, zero, f) for f in np.linspace(0,1,N)]
        colors = lows + highs[::-1]

        return ListedColormap(colors)
    # ====================================================================


    @staticmethod
    def _find_nearest(array, value):
        return (np.abs(np.array(array) - value)).argmin()
    # ====================================================================


# ====================================================================
# ====================================================================





