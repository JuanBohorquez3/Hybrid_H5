"""
A variety of functions useful for plotting data
"""
import pandas as pd
from numpy import *
import matplotlib.pyplot as plt
from typing import Tuple

from DataH5 import fold_to_nd, ivar_step_size


def default_plotting(iterations: pd.DataFrame, data: ndarray, data_ers: ndarray, shots: int = None):
    """
    Plots data in to_plot the default manner, based on the number of independent variables present
    in iterations

    Default plotting modes depend on number or independent variables:
        0 independent variables : prints value of data[0,shot] and data_er[0,shot] for each shot
        1 independent variables : plots values of data[:,shot] and data_er[:,shot] for each shot
        2 independent variables : shows color map image of data[:,shot] for each shot as folded by
            DataH5.fold_to_nd()
        3 or more independent variables : print a warning for the user to use of make a purpose-made
            plot
    Args:
        iterations: DataFrame of iterations and the values of independent variables during that
            iteration
        data: ndarray of data to be plotted. Must be indexed [iteration,shot] with dimensions:
            (iterations, shots)
        data_ers: ndarray of uncertainty in data. Must be indexed [iteration,shot] with dimensions:
            (iterations, shots)
        shots: number of shots taken per measurement. If not specified only the first shot is
            plotted
    """
    shots = 1 if shots is None else shots
    if len(iterations.keys()) == 1:
        for shot in range(shots):
            print(f"shot {shot} rate : {data[0, shot]} +/- {data_ers[0, shot]}")
    elif len(iterations.keys()) == 2:
        independent_variable = list(iterations.keys())[1]
        xlin = iterations[independent_variable]
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        for shot in range(shots):
            ax.errorbar(xlin, data[:, shot], yerr=data_ers[:, shot], label=f"Shot {shot}")
        ax.legend()
        ax.set_ylabel("Counts in ROI")
        ax.set_xlabel(independent_variable)
        fig.show()
    elif len(iterations.keys()) == 3:
        fig, axarr = plt.subplots(1, 2, figsize=(2 * 5, 5))
        iVars = [key for key in iterations if key != 'iteration']
        pix_size_x, pix_size_y = ivar_step_size(iterations)
        for shot in range(shots):
            means_nd = fold_to_nd(iterations, data[:, shot])
            im = axarr[shot].imshow(means_nd, interpolation='none', extent=
            [min(iterations[iVars[1]] - pix_size_x / 2),
             max(iterations[iVars[1]] + pix_size_x / 2),
             max(iterations[iVars[0]] + pix_size_y / 2),
             min(iterations[iVars[0]] - pix_size_y / 2)
             ])
            fig.colorbar(im, ax=axarr[shot], use_gridspec=True, shrink=.7)
            axarr[shot].set_xlabel(iVars[1])
            axarr[shot].set_ylabel(iVars[0])
            axarr[shot].set_title(f"Shot {shot}")
            axarr[shot].set_xticks(round_(iterations[iVars[1]], 2), )
            axarr[shot].set_yticks(round_(iterations[iVars[0]], 2))
        fig.tight_layout()
        fig.suptitle("Images of mean values")
        fig.show()
    else:
        print("many axes, look for purpose made cells")