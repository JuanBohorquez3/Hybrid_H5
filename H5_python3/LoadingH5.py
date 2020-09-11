"""
Loads data from results file into memory
"""
import h5py
from typing import List, Dict, Any
from collections import OrderedDict
from numpy import *
import pandas as pd


def get_iteration_ivars(iteration, *ivar_names: str) -> Dict[str, Any]:
    """
    Gets the values of independent variables for the iteration that was passed in

    Args:
        iteration: h5py data group corresponding to an iteration
        *ivar_names: list of names of independent variables
    Returns:
        map of ivar_names to their values
    """
    return {name: iteration[f"variables/{name}"][()] for name in ivar_names}


def get_indep_vars(results_file) -> OrderedDict[str, List[Any]]:
    """
    Finds the independent variables that were varied this experiment and puts them in a dictionary

    Args:
        results_file : h5py File corresponding to the relevant results file

    Returns:
        dictionary mapping varied independent variable names to the values these variables took.
        Keys are sorted alphabetically
    """

    indep_vars = {}
    if len(results_file['iterations']) > 1:
        for variable in results_file['settings/experiment/independentVariables'].items():
            values = eval(variable[1]['function'][()])
            # print(f"{variable[0]}\n\t{values}")
            if iterable(values):
                indep_vars.update({variable[0]: array(values)})

    return OrderedDict(sorted(indep_vars))


def make_iterations_df(h5file, iVars: List[str]) -> pd.DataFrame:
    """
    Makes a dataframe in which the columns are labeled by the names of the varied independent
    variables from this experiment, and 'iteration'.

    Used to map 'iteration' (iteration number)
    to the values of the independent variables during that iteration
    Args:
        h5file : h5py File corresponding to the results file for this experiment
        iVars : list of independent variables that were varied this experiment

    Returns:
        data frame where 'iteration' number is tabulated with values of independent variables
    """

    iterations = pd.DataFrame(columns=["iteration"] + iVars)
    for iteration in h5file['iterations'].items():
        i = int(iteration[0])
        ivar_vals = get_iteration_ivars(iteration[1], *iVars)
        ivar_vals.update({"iteration": i})
        iterations = iterations.append(pd.DataFrame(ivar_vals, index=[i]))
    return iterations.sort_index()

def fold_to_nd(iterations: pd.DataFrame, data_array: array = None) -> ndarray:
    """
    Folds data array into an ndarray conveniently shaped
    Args:
        iterations : ndarray of experiment data. Indexed by iteration with corresponding values of
            independent variables filling out the other columns
        data_array : 1D array with data from each iteration. Should be indexed [iteration,...]
            if none, an array of iteration numbers is returned
    Returns:
        the folded data_array. Indexed
            [independent_variable_step1,independent_variable_step2,...]
    """