"""
A variety of functions useful for plotting data
"""
import pandas as pd
from numpy import *
import matplotlib.pyplot as plt
from typing import Tuple

from Iterations import Iterations


def _fix_nd_indexing(data: ndarray):
    """
    Fix indexing for a data array that can be indexed [iteration] instead of [iteration, shot]
    Args:
        data : 1D data array indexed [iteration]

    Returns:
        fixed_data : 2D data array indexed [iteration,shot] where the shot axis is of length 1
    """
    fixed_data = zeros((len(data),1))
    fixed_data[:, 0] = data
    return fixed_data


def default_plotting(
        iterations: Iterations,
        data: ndarray,
        data_error: ndarray = None,
        shots: int = 1,
        description: str = "value",
        figsize: Tuple = None,
        **kwargs
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
        data: ndarray of data to be plotted. Can be indexed [iteration,shot] or [iteration] if
            shots = 1.
        data_error: ndarray of uncertainty in data. Must be indexed [iteration,shot] or
            [iteration] if shots = 1
        shots: number of shots taken per measurement. If not specified only the first shot is
            plotted
        description: description of data to be plotted (used for plot title)
        figsize: size of the plot to be made. In the case of 2D scans figsize will be changed to (fs[0]*shots, fs[1]). Defaults are:
            0 ivs : Irrelevant
            1 ivs : (6,6)
            2 ivs : (shots * 5, 5)
        **kwargs: kwargs to be passed to relevant plotting function. In 1D scans it's errorbar() in 2D functions its imshow()
    """
    # Set default
    data_error = zeros(data.shape, dtype=float) if data_error is None else data_error

    # Fix data and data_error indexing if necessary
    if data.shape != data_error.shape:
        raise ValueError("data and data error must have the same shape")
    if len(data.shape) == 1:
        data = _fix_nd_indexing(data)
        data_error = _fix_nd_indexing(data_error)

    if len(iterations.keys()) == 1:
        for shot in range(shots):
            print(f"shot {shot} {description} : {data[0, shot]:.3f} +/- {data_error[0, shot]:.3f}")
    elif len(iterations.keys()) == 2:
        independent_variable = list(iterations.keys())[1]
        if figsize is None:
            figsize = (6, 5)
        xlin = iterations[independent_variable]
        # Sort data to match ordering of iterations dataframe
        data = data[array(iterations['iteration'], dtype=int)]
        data_error = data_error[array(iterations['iteration'], dtype=int)]
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        for shot in range(shots):
            ax.errorbar(xlin, data[:, shot], yerr=data_error[:, shot], label=f"Shot {shot}")
        ax.legend()
        ax.set_ylabel(description)
        ax.set_xlabel(independent_variable)
        fig.tight_layout()
        fig.show()
    elif len(iterations.keys()) == 3:
        if figsize is None:
            figsize = (shots * 5, 5)
        else:
            figsize = (shots * figsize[0], figsize[1])
        fig, axarr = plt.subplots(1, shots, figsize=figsize)
        extent = [
            min(iterations[iterations.ivars[1]] - iterations._step_sizes[1] / 2),  # left
            max(iterations[iterations.ivars[1]] + iterations._step_sizes[1] / 2),  # right
            max(iterations[iterations.ivars[0]] + iterations._step_sizes[0] / 2),  # bottom
            min(iterations[iterations.ivars[0]] - iterations._step_sizes[0] / 2)   # top
        ]
        if shots == 1:
            axarr = [axarr]
        for shot in range(shots):
            means_nd = iterations.fold_to_nd(data[:, shot])
            im = axarr[shot].imshow(means_nd, interpolation='none', aspect='auto', extent=extent, **kwargs)
            fig.colorbar(im, ax=axarr[shot], use_gridspec=True, shrink=.7)
            axarr[shot].set_xlabel(iterations.ivars[1])
            axarr[shot].set_ylabel(iterations.ivars[0])
            if shots - 1:
                axarr[shot].set_title(f"Shot {shot}")
            axarr[shot].set_xticks(round_(array(iterations[iterations.ivars[1]]).astype(float), 2))
            axarr[shot].set_yticks(round_(array(iterations[iterations.ivars[0]]).astype(float), 2))
        # fig.tight_layout()
        fig.suptitle(description)
        fig.tight_layout()
        fig.show()
    else:
        print("too many axes, look for purpose made cells")


def iterate_plot_2D(
        iterations: Iterations,
        data: ndarray,
        data_error: ndarray = None,
        x_ivar: str = None,
        description: str = "arb",
        shots: int = 2,
        **kwargs
):
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
    Returns:
        fig, axarr: figure and axis array that was plotted
    """
    # Set default
    data_error = zeros(data.shape, dtype=float) if data_error is None else data_error

    # Fix data and data_error indexing if necessary
    if data.shape != data_error.shape:
        raise ValueError("data and data error must have the same shape")
    if len(data.shape) == 1:
        data = _fix_nd_indexing(data)
        data_error = _fix_nd_indexing(data_error)

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
            axarr[i].errorbar(x_vals, data_vals, yerr=data_ers, label=f"Shot {shot}", **kwargs)
            axarr[i].set_xlabel(x_ivar)
            axarr[i].set_ylabel(description)
            axarr[i].set_title(f"{y_ivar} = {y_val}")
            axarr[i].legend()
    fig.tight_layout()
    fig.show()

    return fig, axarr


# misc utility functions  --------------------------------------------------------------------------
def expand_iter_array(iterations, iter_array, no_measurements, no_shots):
    """
    Expands an array of len() == len(iterations) to an ndarray of the shape of shot-by-shot data.

    An iter array is an array of shape = (len(iteration),), and is indexed [iteration].
        The assumption being that it's an array containing iteration-by-iteration data.
    An array with shot-by-shot data is indexed [iteration,measurement,shot]
    """
    if len(iter_array) != len(iterations):
        raise ValueError("Iter arrays must have length of iterations")
    fixed_array = ones((len(iterations), no_measurements, no_shots))
    for i, row in iterations.iterrows():
        iteration = row['iteration']
        fixed_array[iteration] = iter_array[iteration] * fixed_array[iteration]
    return fixed_array