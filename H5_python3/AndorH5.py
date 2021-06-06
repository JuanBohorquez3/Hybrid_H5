"""
Workhorse file to perform analysis on data taken by Andor Cameras using CSPY
Author : Juan Bohorquez
Created on : 06/04/2021
Last Modified : 06/04/2021
"""

import h5py
import os
import numpy as np
import warnings
from typing import Tuple

from HamamatsuH5 import HMROI


def load_data(
        results_file: h5py.File,
        roi: HMROI
) -> np.array:
    """
    Loads data from an Andor camera into a numpy array
    
    results are indexed as follows
    > results = array[iterations,measurements,shots,horizontal_pixels, vertical_pixels]
    Args:
        results_file: h5file object corresponding to results.hdf5 file
        roi: region of interest from which to extract pixel data

    Returns:
        5D numpy array holding all of the data taken by the hamamatsu during the experiment
        indexed [iteration,measurement,shot,horizontal_pixel,vertical_pixel]
    """

    num_its = len(results_file['iterations'])
    measurements = results_file['settings/experiment/measurementsPerIteration'][()] + 1
    shots_per_measurement = 1

    andr_pix = np.zeros(
        (num_its, measurements, shots_per_measurement, roi.bottom - roi.top, roi.right - roi.left,),
        dtype=int
    )

    for iteration, i_group in results_file['iterations'].items():
        # print(f"iteration : {iteration} : {type(iteration)}")
        for measurement, m_tup in enumerate(i_group['measurements'].items()):
            m_group = m_tup[1]
            # print(f"\tmeasurement : {measurement} : {type(measurement)}")
            for shot, s_group in m_group['data/Andor_1026/shots'].items():
                try:
                    # print(f"\t\tshot : {shot} : {type(shot)}")
                    andr_pix[int(iteration), int(measurement), int(shot)] = s_group[()][roi.slice]
                except IndexError as e:
                    warnings.warn(
                        f"{e}\n iteration : {iteration} measurement : {measurement} shot {shot}"
                    )
                    continue
                except ValueError as ve:
                    warnings.warn(
                        f"{ve}\n iteration : {iteration} measurement : {measurement} shot {shot}"
                    )

    return andr_pix