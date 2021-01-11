"""
Workhorse file to perform analysis on data taken by counters using CSPY
Author : Juan Bohorquez
Created on : 09/02/2020
Last Modified : 09/02/2020
"""

import h5py
import os
import numpy as np
import warnings
from typing import Tuple


def load_data(
        results_file: h5py.File,
        drop_bins: int,
        ro_bins: int,
        shots: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from a counter instrument into a numpy array

    results are indexed as follows
    > results = array[iteration, measurement, shot]
    Args:
        results_file: h5file object corresponding to the results.hdf5 file being analyzed
        drop_bins: number of bins to drop at the start of each shot
        ro_bins: number of readout bins expected in each shot
        shots: number of shots taken in the experiment. Default is 2

    Returns:
        (binned_counter_data, shot_counter_data)
            binned_counter_data : 4D numpy array holding the counter data taken by the given counter
                during the experiment. Each entry corresponds to the number of counts detected
                during a binning period
                indexed [iteration, measurement, shot, ro_bin]
            shot_counter_data : a 3D numpy array holding the counter data taken by the given counter
                during the experiment. Each entry corresponds to the number of counts detected
                during a shot. counts detected by the first drop_bins bins in a shot are discarded
                from this array
                shot_counter_data = binned_counter_data[..., drop_bins:].sum(3)
                indexed [iteration, measurement, shot]
    """
    num_its = len(results_file['iterations'])
    measurements = results_file['settings/experiment/measurementsPerIteration'][()]+1
    if shots is None:
        shots_per_measurement = int(
            results_file['/settings/experiment/LabView/camera/shotsPerMeasurement/function'][()])
    else:
        shots_per_measurement = shots

    binned_counter_data = np.ones(
        (num_its, measurements, shots_per_measurement, drop_bins + ro_bins),
        dtype=int
    )*-1

    filter_offset = check_n_filter(results_file)

    for iteration, i_group in results_file['iterations'].items():
        # print(f"iteration : {iteration} : {type(iteration)}")
        for measurement, m_group in i_group['measurements'].items():
            # load data, a 1D array of counter data from the measurement, each entry corresponds to
            # the measured counts in a given binning period
            raw_data = np.array(m_group['data/counter/data'][()])[0]

            # Loose cables or improper triggering can lead to some lost bins. We simply do not write
            # this data to our data arrays
            if raw_data.shape[0] != shots_per_measurement*(drop_bins+ro_bins):
                inds = {
                    "iteration": iteration,
                    "measurement": measurement
                }
                ind_msg = ",".join([f"{var} : {val}" for var, val in inds.items()])
                print(
                    f"Invalid data shape in hdf5 file. raw_data.shape = {raw_data.shape}" +
                    f"\n\tlocation : {ind_msg}"
                )
                continue
            for shot in range(shots_per_measurement):
                tot_bins = drop_bins + ro_bins
                iteration = int(iteration)
                measurement = int(measurement)
                if measurement < filter_offset:
                    print(f"measurement {measurement} has been dropped due to fist-N-filter. First {filter_offset} being dropped")
                    continue
                try:
                    binned_counter_data[iteration, measurement - filter_offset, shot, :] = raw_data[shot*tot_bins: (shot + 1) * tot_bins]
                # Catch issues reshaping raw_data and omit that shot from binned_counter_data
                except (IndexError, ValueError) as e:
                    inds = {
                        "iteration": iteration,
                        "measurement": measurement,
                        "measurement - filter_offset": measurement - filter_offset,
                        "shot": shot,
                        "tot_bins": tot_bins,
                        "raw_data.shape": raw_data.shape}
                    print(iteration)
                    ind_msg = ",".join([f"{var} : {val}" for var, val in inds.items()])
                    print(f"Warning. Error reading data for {ind_msg}\n {e}")
                    continue

    shot_counter_data = binned_counter_data[..., drop_bins:].sum(3)
    # Fix holes in data by replacing count data with the minimum acquired value
    # (registers as 0 atoms)
    min_at = min(shot_counter_data[np.where(shot_counter_data > -1)])
    shot_counter_data[np.where(shot_counter_data < 0)] = min_at

    return binned_counter_data, shot_counter_data


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


def check_n_filter(results_file: h5py.File) -> int:
    """
    Checks the filters settings. Returns an offset if the "Drop First N Measurements Filter" is
    enabled. Returns the number of measurements being dropped at the beginning
    Args:
        results_file: h5file object corresponding to the results.hdf5 file being analyzed

    Returns:
        Offset: Number of measurements being dropped at the beginning of each iteration. If filter
            is not enabled, returns 0.
    """

    enabled = results_file["settings/experiment/first_measurements_filter/enable"][()]

    return enabled * results_file["settings/experiment/first_measurements_filter/N"][()]