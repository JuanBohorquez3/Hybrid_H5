"""
A variety of functions useful for plotting data
"""
import pandas as pd
from numpy import *
import matplotlib.pyplot as plt
from typing import Tuple, Union, List

from Iterations import Iterations


def get_var(its: Iterations, variable: str = None) -> Tuple[str, int]:
    """
    Get variable name and and index from iterations object
    """
    if variable is not None:
        if variable not in its.vars:
            raise KeyError(
                f"{variable} is not an independent or dependent variable of this experiment.")
        var_ind = [i for i, v in enumerate(its.vars) if v == variable][0]
    else:
        # Prompt the user to input which independent variable is on the x-axis of each graph
        ms = "Which Independent variable or dependent variable is on the x-axis? :\n"
        for i, v in enumerate(its.vars):
            ms += f"\t{i}) : {v}\n"
        while True:
            try:
                var_ind = int(input(ms))
                variable = its.vars[var_ind]
            except (ValueError, IndexError):
                print(f"Not a valid input, input an int between 0 and {len(its.vars) - 1}")
                continue
            else:
                break
    return variable, var_ind

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
        shots: Union[int, List] = 1,
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
        num_shots: number of shots taken per measurement. If not specified only the first shot is
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

    if type(shots) is int:
        num_shots = shots
    else:
        num_shots = len(shots)

    if len(iterations.ivars) == 0:
        for shot in range(num_shots):
            print(f"shot {shot} {description} : {data[0, shot]:.3f} +/- {data_error[0, shot]:.3f}")
        return None, None
    elif len(iterations.ivars) == 1:
        independent_variable = list(iterations.keys())[1]
        if figsize is None:
            figsize = (6, 5)
        xlin = iterations[independent_variable]
        # Sort data to match ordering of iterations dataframe
        data = data[array(iterations['iteration'], dtype=int)]
        data_error = data_error[array(iterations['iteration'], dtype=int)]
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        for shot in range(num_shots):
            if type(shots) is int:
                label = f"Shot = {shot}"
            else:
                label = f"{shots[shot]}"
            ax.errorbar(xlin, data[:, shot], yerr=data_error[:, shot], label=label, **kwargs)
        ax.legend()
        ax.set_ylabel(description)
        ax.set_xlabel(independent_variable)
        fig.tight_layout()
        fig.show()
        return fig, ax
    elif len(iterations.ivars) == 2:
        if figsize is None:
            figsize = (num_shots * 5, 5)
        else:
            figsize = (num_shots * figsize[0], figsize[1])
        fig, axarr = plt.subplots(1, num_shots, figsize=figsize)
        extent = [
            min(iterations[iterations.ivars[1]] - iterations._step_sizes[1] / 2),  # left
            max(iterations[iterations.ivars[1]] + iterations._step_sizes[1] / 2),  # right
            max(iterations[iterations.ivars[0]] + iterations._step_sizes[0] / 2),  # bottom
            min(iterations[iterations.ivars[0]] - iterations._step_sizes[0] / 2)   # top
        ]
        if num_shots == 1:
            axarr = [axarr]
        for shot in range(num_shots):
            means_nd = iterations.fold_to_nd(data[:, shot])
            im = axarr[shot].imshow(means_nd, interpolation='none', aspect='auto', extent=extent, **kwargs)
            fig.colorbar(im, ax=axarr[shot], use_gridspec=True, shrink=.7)
            axarr[shot].set_xlabel(iterations.ivars[1])
            axarr[shot].set_ylabel(iterations.ivars[0])
            if num_shots - 1:
                if type(shots) is int:
                    title = f"Shot = {shot}"
                else:
                    title = f"{shots[shot]}"
                axarr[shot].set_title(title)
            #axarr[shot].set_xticks(round_(array(iterations[iterations.ivars[1]]).astype(float), 6))
            #axarr[shot].set_yticks(round_(array(iterations[iterations.ivars[0]]).astype(float), 6))
        # fig.tight_layout()
        fig.suptitle(description)
        fig.tight_layout()
        fig.show()
        return fig, axarr
    else:
        print("too many axes, look for purpose made cells")

def iterate_plot_2D(
        iterations: Iterations,
        data: ndarray,
        data_error: ndarray = None,
        x_var: str = None,
        it_var: str = None,
        description: str = "arb",
        shots: Union[int, List] = 2,
        axsize=None,
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
        x_var : variable to place on x-axis. If not specified the user is prompted to
            choose one when this code is run
        it_var : variable over which plots iterate. If not specified the user is prompted to choose one when code is run
        description : description of data, put on y-axis of plots
        shots : number of shots specified in data. If a list, len is taken as number of shots
        axsize : size of individual axes
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

    x_var, x_var_ind = get_var(iterations, x_var)

    it_var, it_var_ind = get_var(iterations, it_var)

    # print(f"x_var(_ind) = {x_var, x_var_ind}")
    # print(f"it_var(_ind) = {it_var, it_var_ind}")
    if it_var not in iterations.independent_variables:
        raise ValueError("iterated variable must be an independent variable")

    if it_var == x_var:
        raise ValueError("iterated variable must be distinct from x_var")

    if type(shots) is int:
        num_shots = shots
    else:
        num_shots = len(shots)

    plots = len(iterations.independent_variables[it_var])
    if axsize is None:
        figsize = (6, 4 * plots)
    else:
        figsize = (axsize[0], axsize[1]*plots)
    fig, axarr = plt.subplots(plots, 1, figsize=figsize)

    for shot in range(num_shots):
        data_nd = iterations.fold_to_nd(data[:, shot])
        error_nd = iterations.fold_to_nd(data_error[:, shot])
        # print(it_var_ind)
        if it_var_ind:
            # print("transposing data_nd")
            data_nd = data_nd.T
            error_nd = error_nd.T

        for i, data_vals in enumerate(data_nd):
            data_ers = error_nd[i]
            if x_var in iterations.independent_variables:
                x_vals = sorted(iterations.independent_variables[x_var])  # values taken on the x-axis
            else:
                x_indep = iterations.ivars[not it_var_ind]
                x_len = len(iterations.independent_variables[x_indep])
                if it_var_ind:
                    x_vals = iterations[x_var][i*x_len:(i+1)*x_len]
                else:
                    x_vals = iterations[x_var][i::len(iterations.independent_variables[it_var])]
                # print(f"i, data_vals, len(data_vals) = {i}, {data_vals}, len(data_vals)")
                # print(x_indep, x_len, x_vals)
                # print(f"len(x_vals) = {len(x_vals)}")
            # value of the iterated variable for this run through the loop
            it_val = sorted(iterations.independent_variables[it_var])[i]
            label = shot if type(shots) is int else shots[shot]
            axarr[i].errorbar(x_vals, data_vals, yerr=data_ers, label=label, **kwargs)
            axarr[i].set_xlabel(x_var)
            axarr[i].set_ylabel(description)
            axarr[i].set_title(f"{it_var} = {it_val}")
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