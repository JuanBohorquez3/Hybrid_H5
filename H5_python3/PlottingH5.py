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
        data_error: ndarray = None,
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
        data_error: ndarray of uncertainty in data. Must be indexed [iteration,shot] with dimensions:
            (iterations, shots)
        shots: number of shots taken per measurement. If not specified only the first shot is
            plotted
        description: description of data to be plotted (used for plot title)
    """
    # Set default
    data_error = zeros(size(data), dtype=float) if data_error is None else data_error

    if len(iterations.keys()) == 1:
        for shot in range(shots):
            print(f"shot {shot} {description} : {data[0, shot]} +/- {data_error[0, shot]}")
    elif len(iterations.keys()) == 2:
        independent_variable = list(iterations.keys())[1]
        xlin = iterations[independent_variable]
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        for shot in range(shots):
            ax.errorbar(xlin, data[:, shot], yerr=data_error[:, shot], label=f"Shot {shot}")
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


def iterate_plot_2D(
        iterations: Iterations,
        data: ndarray,
        data_error: ndarray = None,
        x_ivar: str = None,
        description: str = "arb",
        shots: int = 2):
    """
    plot a 2D experiment by plotting a line graph for each value of one of the independent
    variables.

    the values of x_ivar are places on the x-axis of each plot.
    Args:
        iterations : Iterations object containing relevant experiment information
        data : data array, indexed [iteration,shot]
        data_error : uncertainty in data, indexed [iteration,shot]
        x_ivar : independent variable to place on x-axis. If not specified the user is prompted to
            choose one when this code is run
        description : description of data, put on y-axis of plots
        shots : number of shots specified in data
    """
    # TODO : create function to allow data indexed [iteration] when shots == 1
    if len(iterations.ivars) != 2:
        raise ValueError(
            "This plot only works to plot data from an experiment with two independent variables")

    if x_ivar is not None:
        if x_ivar not in iterations.ivars:
            raise KeyError(
                f"{x_ivar} is not an independent variable of this experiment.")
        x_ivar_ind = int(iterations.ivars[1] == x_ivar)
    else:
        # Prompt the user to input which independent variable is on the x-axis of each graph
        ms = "Which Independent variable is on the x-axis? :\n"
        for i, ivar in enumerate(iterations.ivars):
            ms += f"\t{i}) : {ivar}\n"
        while True:
            try:
                x_ivar_ind = int(input(ms))
                x_ivar = iterations.ivars[x_ivar_ind]
            except (ValueError, IndexError):
                print(f"Not a valid input, input an int between 0 and {len(iterations.ivars) - 1}")
                continue
            else:
                break

    y_ivar = iterations.ivars[not (x_ivar_ind)]

    plots = len(iterations.independent_variables[y_ivar])
    fig, axarr = plt.subplots(plots, 1, figsize=(6, 4 * plots))
    for shot in range(shots):
        data_nd = iterations.fold_to_nd(data[:, shot])
        error_nd = iterations.fold_to_nd(data_error[:, shot])
        if not x_ivar_ind:  # Transpose the data array if the index is right, for convenience
            data_nd = data_nd.T
            error_nd = error_nd.T

        x_vals = sorted(iterations.independent_variables[x_ivar])  # values taken on the x-axis
        for i, data_vals in enumerate(data_nd):
            data_ers = error_nd[i]
            # value of the y_ivar for this run through the loop
            y_val = sorted(iterations.independent_variables[y_ivar])[i]
            axarr[i].errorbar(x_vals, data_vals, yerr=data_ers, label=f"Shot {shot}")
            axarr[i].set_xlabel(x_ivar)
            axarr[i].set_ylabel(description)
            axarr[i].set_title(f"{y_ivar} = {y_val}")
            axarr[i].legend()
    fig.tight_layout()
    fig.show()
