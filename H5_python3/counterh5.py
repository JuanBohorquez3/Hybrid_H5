"""
Workhorse file to perform analysis on data taken by counters using CSPY
Author : Juan Bohorquez
Created on : 09/02/2020
Last Modified : 09/02/2020
"""

import h5py
import os


def open_results(path: str, results_file: str = "results.hdf5") -> h5py.File:
    """
    Opens the results of an experiment specified by the path argument.

    Args:
        path: valid path to the folder containing the results file of interest
        results_file: filename of results file

    Raises:
        ValueError : When the combination of path and file provided do not lead to a valid hdf5 file

    Returns:
        opened h5py.File object
    """
    full_path = os.path.join(path, results_file)
    if os.path.isfile(full_path) and results_file[-4:] == "hdf5":
        h5_file = h5py.File(full_path, mode="r")
        return h5_file
    else:
        raise ValueError(f"{full_path} is not a valid hdf5 file")

