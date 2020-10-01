"""
A variety of functions useful for plotting data
"""
import pandas as pd
from numpy import *
import matplotlib.pyplot as plt

from Iterations import Iterations

def default_plotting(
        iterations: Iterations,
        data: ndarray,
        data_ers: ndarray = None,
        shots: int = 1,
        description: str = "value"
):
    """
    Plots data in data the default manner, based on the number of independent variables present
    in iterations

    Default plotting modes depend on number or independent variables:
        0 independent variables : prints value of data[0,shot] and data_er[0,shot] for each shot
        1 independent variables : plots values of data[:,shot] and data_er[:,shot] for each shot
        2 independent variables : shows color map image of data[:,shot] for each shot as folded by
            Iterations.fold_to_nd()
        3 or more independent variables : print a warning for the user to use of make a purpose-made
            plot
    Args:
        iterations: Iterations class holding this experiment's relevant data
        data: ndarray of data to be plotted. Must be indexed [iteration,shot] with dimensions:
            (iterations, shots)
        data_ers: ndarray of uncertainty in data. Must be indexed [iteration,shot] with dimensions:
            (iterations, shots)
        shots: number of shots taken per measurement. If not specified only the first shot is
            plotted
        description: description of data to be plotted (used for plot title)
    """
    # Set default
    data_ers = zeros(size(data), dtype=float) if data_ers is None else data_ers

    if len(iterations.keys()) == 1:
        for shot in range(shots):
            print(f"shot {shot} {description} : {data[0, shot]} +/- {data_ers[0, shot]}")
    elif len(iterations.keys()) == 2:
        independent_variable = list(iterations.keys())[1]
        xlin = iterations[independent_variable]
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        for shot in range(shots):
            ax.errorbar(xlin, data[:, shot], yerr=data_ers[:, shot], label=f"Shot {shot}")
        ax.legend()
        ax.set_ylabel(description)
        ax.set_xlabel(independent_variable)
        fig.show()
    elif len(iterations.keys()) == 3:
        fig, axarr = plt.subplots(1, 2, figsize=(2 * 5, 5))
        extent = [
            min(iterations[iterations.ivars[1]] - iterations.step_sizes[1]/2),  # left
            max(iterations[iterations.ivars[1]] + iterations.step_sizes[1]/2),  # right
            max(iterations[iterations.ivars[0]] + iterations.step_sizes[0]/2),  # bottom
            min(iterations[iterations.ivars[0]] - iterations.step_sizes[0]/2)   # top
        ]
        for shot in range(shots):
            means_nd = iterations.fold_to_nd(data[:, shot])
            im = axarr[shot].imshow(means_nd, interpolation='none', aspect='auto', extent=extent)
            fig.colorbar(im, ax=axarr[shot], use_gridspec=True, shrink=.7)
            axarr[shot].set_xlabel(iterations.ivars[1])
            axarr[shot].set_ylabel(iterations.ivars[0])
            axarr[shot].set_title(f"Shot {shot}")
            axarr[shot].set_xticks(round_(iterations[iterations.ivars[1]], 2), )
            axarr[shot].set_yticks(round_(iterations[iterations.ivars[0]], 2))
        fig.tight_layout()
        fig.suptitle(description)
        fig.show()
    else:
        print("many axes, look for purpose made cells")