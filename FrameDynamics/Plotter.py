"""
Date: 2022/12
Author: Jens D. Haller
Mail: jens.haller@kit.edu / jhaller@gmx.de
Institution: Karlsruhe Institute of Technology

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import ImageGrid


class Plotter():
    
    def save(self, name:str, dpi=None) -> None:
        self.fig.savefig(name, dpi=dpi)
    # ====================================================================

    def show(self) -> None:
        plt.show()
    # ====================================================================
    
    def close(self) -> None:
        plt.close("all")
    # ====================================================================


class Grid(Plotter):

    def __init__(self, n):
        self.n = n
        self.fig = plt.figure(figsize=(10, 0.5+n*1.5))
        self.ax = ImageGrid(self.fig, 111,         
                  nrows_ncols=(n,1),
                  axes_pad=0.23,
                  share_all=True,
                  aspect=False)
    # ====================================================================

    def plot1D(self, X, Y):

        for i in range(self.n):
            self.ax[i].plot(X[i], Y[i], color='#80000f', lw=2.)
            self.ax[i].fill_between(X[i], Y[i], color='#cc000f')
            self.ax[i].set_ylabel("k(t)", rotation=0, size=14)
            self.ax[i].yaxis.set_label_coords(-0.05,0.85)
            self.ax[i].plot([X[i][0],X[i][-1]],[1,1],"k--",lw=0.5)
            self.ax[i].plot([X[i][0],X[i][-1]],[0,0],"k",lw=1)
            self.ax[i].plot([X[i][0],X[i][-1]],[-1,-1],"k--",lw=0.5)
            
        self.ax[0].set_xlim([X[i][0], X[i][-1]])
    # ====================================================================

    def set_labels(self, labels, AHT):

        for i in range(self.n):
            self.ax[i].text(-0.1,0.5, labels[i], size=18, \
                            transform=self.ax[i].transAxes, ha="center")
            L = "({}%)".format(AHT[i])
            self.ax[i].text(-0.1,0.31, L, size=12, \
                            transform=self.ax[i].transAxes, ha="center")
    # ====================================================================

    def set_xaxis(self, label):
        self.ax[self.n-1].set_xlabel(label, fontsize=15)
    # ====================================================================



class Subplot(Plotter):

    def __init__(self, spinY:str, spinX:str, size=(10, 9)) -> None:
        
        self.fig, self.ax = plt.subplots(3, 3, figsize=size, sharex=True, \
                                         sharey=True, constrained_layout=True)

        self.labels = [r"$2{}_x{}_x$".format(spinY, spinX),
                       r"$2{}_x{}_y$".format(spinY, spinX),
                       r"$2{}_x{}_z$".format(spinY, spinX),
                       r"$2{}_y{}_x$".format(spinY, spinX),
                       r"$2{}_y{}_y$".format(spinY, spinX),
                       r"$2{}_y{}_z$".format(spinY, spinX),
                       r"$2{}_z{}_x$".format(spinY, spinX),
                       r"$2{}_z{}_y$".format(spinY, spinX),
                       r"$2{}_z{}_z$".format(spinY, spinX),
                      ]
    # ====================================================================
    
    def plot1D(self, X, Y, **kwargs) -> None:

        for i in range(9):
            self.ax[i//3, i%3].plot([X[0], X[-1]], [0, 0], "k--", lw=1)
            self.ax[i//3, i%3].plot(X, Y[i], lw=2.5, color="#1425a4", **kwargs)
            self.ax[i//3, i%3].text(0.05, 0.85, self.labels[i], \
                            transform=self.ax[i//3, i%3].transAxes, size = 20)

        self.ax[0,0].set_xlim([X[0], X[-1]])
    # ====================================================================
    
    def plot2D(self, X, Y, Z, vals, levels, **kwargs) -> None:

        # Create colormap and retrieve data for plotting
        RWB = self._getColorMap(levels-1)

        for i in range(9):
            self.cb = self.ax[i//3, i%3].contourf(X, Y, Z[i], \
                      levels=vals, cmap=RWB)
            self.ax[i//3, i%3].text(0.05, 0.85, self.labels[i], \
                        transform=self.ax[i//3, i%3].transAxes, size = 20)
        
        self.fig.colorbar(self.cb, ax=self.ax[0,2])
        self.fig.colorbar(self.cb, ax=self.ax[1,2])
        self.fig.colorbar(self.cb, ax=self.ax[2,2])
    # ====================================================================
    
    def set_xaxis(self, label:str, size=15) -> None:
        self.ax[2,0].set_xlabel(label, size=size)
        self.ax[2,1].set_xlabel(label, size=size)
        self.ax[2,2].set_xlabel(label, size=size)
    # ====================================================================
    
    def set_yaxis(self, label:str, size:float=15) -> None:
        self.ax[0,0].set_ylabel(label, size=size)
        self.ax[1,0].set_ylabel(label, size=size)
        self.ax[2,0].set_ylabel(label, size=size)
    # ====================================================================
    
    def constrainedLayout(self) -> None:
        self.fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025,
            hspace=0.025, wspace=0.025)
    # ====================================================================
    
    @staticmethod
    def _interpColors(V, Z, f):
        """ 
        V corresponds to a value
        Z corresponds to what is set to "zero"
        f is scaling factor
        """

        return (V[0]+f*(Z[0]-V[0]), V[1]+f*(Z[1]-V[1]), V[2]+f*(Z[2]-V[2]) )
    # ====================================================================

    def _getColorMap(self, n):

        high = (20/255, 50/255, 180/255)
        zero = (1, 1, 1)
        low =  (160/255, 0, 0)

        highs = [self._interpColors(high,zero,f) for f in np.linspace(0,1,n)]
        lows = [self._interpColors(low, zero, f) for f in np.linspace(0,1,n)]
        colors = lows + highs[::-1]

        return ListedColormap(colors)
    # ====================================================================
