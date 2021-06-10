import h5py
import os
import numpy as np
import warnings
from numpy import sqrt, diag
from scipy import optimize as opt


def get_ivar_vals(experiment):
    return [vals for vals in experiment.ivarValueLists if len(vals) > 1][0]


def load_data(
        results_file,
        cut_off,
        roi,
        experiment
):
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

    current_iteration = experiment.iteration
    total_iterations = experiment.totalIterations
    opt_iteration = int(current_iteration/total_iterations)

    measurements = len(results_file['/iterations/0/measurements'].items())
    shots_per_measurement = 1

    roi_slice = np.s_[roi["top"]:roi["bottom"], roi["left"]:roi["right"]]

    print(
        "current_iteration : {}, total_iterations : {}, opt_iteration : {}".format(
            current_iteration, total_iterations, opt_iteration
        )
    )

    andr_pix = np.zeros(
        (total_iterations, measurements, shots_per_measurement, roi["bottom"] - roi["top"], roi["right"] - roi["left"],),
        dtype=int
    )

    for iteration in range(0, total_iterations):
        # print(f"iteration : {iteration} : {type(iteration)}")
        it = opt_iteration * total_iterations + iteration
        i_group = results_file['iterations/{}'.format(it)]
        print("iteration: {}, it: {}, i_group: {}".format(iteration, it, i_group))
        for measurement, m_tup in enumerate(i_group['measurements'].items()):
            m_group = m_tup[1]
            # print(f"\tmeasurement : {measurement} : {type(measurement)}")
            for shot, s_group in m_group['data/Andor_1026/shots'].items():
                try:
                    # print(f"\t\tshot : {shot} : {type(shot)}")
                    andr_pix[int(iteration), int(measurement), int(shot)] = s_group[()][roi_slice]
                except IndexError as e:
                    warnings.warn(
                        "{}\n iteration : {} measurement : {} shot {}".format(
                            e, iteration, measurement, shot
                        )
                    )
                    continue
                except ValueError as ve:
                    warnings.warn(
                        "{}\n iteration : {} measurement : {} shot {}".format(
                            ve, it, measurement, shot
                        )
                    )

    cut_pix = andr_pix > cut_off
    pass_frac = cut_pix.sum(3).sum(3) / (andr_pix.shape[-1] * andr_pix.shape[-2])

    return pass_frac.mean(1)[:, 0], pass_frac.std(1)[:, 0]


def fit_to_lorenz(x_data, y_data, y_err, guess):
    def lorenz(x, x0, fwhm, a, o):
        """
        Lorenzian function
        Args:
            x: frequency(s) to evaluate function
            x0: center frequency of resonance
            fwhm: full-width-half-max of resonance
            a: amplitude of function
            o: offset of function from 0

        Returns:
            o + a /((x-x0)**2 + (fwhm)**2)
        """
        return o + a / ((x-x0)**2 + (fwhm)**2)

    func = lambda x, x0, fwhm, a, o: lorenz(x, x0, fwhm, a, o)

    try:
        popt, pcov = opt.curve_fit(f=func, xdata=x_data, ydata=y_data, sigma=y_err, p0=guess)
        perr = sqrt(diag(pcov))
    except RuntimeError as rte:
        print("{}:\nFailed to fit results")
        popt, perr = None, None

    return popt, perr