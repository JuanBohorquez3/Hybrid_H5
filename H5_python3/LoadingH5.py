"""
Loads data from results file into memory
"""
import h5py
from typing import List, Dict, Any
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


def get_indep_vars(results_file) -> Dict[str, List[Any]]:


    indep_vars = {}
    if len(results_file['iterations']) > 1:
        for variable in results_file['settings/experiment/independentVariables'].items():
            values = eval(variable[1]['function'][()])
            # print(f"{variable[0]}\n\t{values}")
            if iterable(values):
                indep_vars.update({variable[0]: list(values)})

    return indep_vars


def make_iterations_df(h5file, iVars: List[str]) -> pd.DataFrame:
    iterations = pd.DataFrame(columns=["iteration"] + iVars)
    for iteration in h5file['iterations'].items():
        i = int(iteration[0])
        ivar_vals = get_iteration_ivars(iteration[1], *iVars)
        ivar_vals.update({"iteration": i})
        iterations = iterations.append(pd.DataFrame(ivar_vals, index=[i]))
    return iterations.sort_index()