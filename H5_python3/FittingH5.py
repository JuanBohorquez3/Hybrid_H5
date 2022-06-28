import math
from numpy import *
from typing import Union, Tuple, Callable, List
from scipy.special import gamma, erf
from scipy import integrate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from Iterations import Iterations
from PlottingH5 import _fix_nd_indexing, get_var


# Probability and Statistics  ----------------------------------------------------------------------
def shot_error(probability: float, samples: int):
    """
    Args:
        probability : number between 0 and 1. Measured probability of an event occurring
        samples : number of samples used to measure the ratio
    Returns:
        shot noise uncertainty. Uncertainty in probability due to statistical variations
    """
    return sqrt(probability*(1-probability)/samples)


# Functions to fit  --------------------------------------------------------------------------------
def iterate_fit_2D(
        iterations: Iterations,
        data: ndarray,
        func: Callable,
        guess: List[float] = None,
        guesses: List[List[float]] = None,
        g_split: Union[int, List[int]] = -1,
        data_error: ndarray = None,
        x_var: str = None,
        it_var: str = None,
        shots: int = 2,
) -> Tuple[ndarray,ndarray]:
    """
    Iterates over a 2D experiment, choosing one independent variable as an x-axis, fitting each
    y-axis to func.


    the values of x_ivar are places on the x-axis of each plot.
    Args:
        iterations : Iterations object containing relevant experiment information
        data : data array, indexed [iteration,shot]
        func : function to fit to
        guess : initial guess for fitting parameters. If None guesses and g_split are used
        guesses : List of initial guesses of fitting parameters. Used in tandem with g_split.
        g_split : List outlining which fits should use which guesses. Use in tandem with guesses argument.
            Use cases based on guesses.shape[0]:
                == 0 : Empty array, throw a ValueError
                == 1 : Same functionality as if guess had been used
                == 2 : guesses[0] used for fits [:g_split] guesses[1] used for fits [g_split:]. If g_split
                    is a list, g_split == g_split[0]
                >= 3 : guesses[i] used for fits [g_split[i-1]:g_split[i]]. If i == 0, g_split[i-1] = 0. If
                    i == guesses.shape[0], g_split[i] = -1. If g_split is not a List, throw a value error
        data_error : uncertainty in data, indexed [iteration,shot]
        x_var : independent variable to place on x-axis. If not specified the user is prompted to
            choose one when this code is run
        shots : number of shots specified in data
    Returns:
        popts, perrs :
            popts : list of fitting parameters. Indexed [y_value_index, shot, parameter]
            perrs : list of uncertainties in fitting parameters.
                Indexed [y_value_index, shot, parameter]
    """
    # Set default
    # data_error = zeros(data.shape, dtype=float) if data_error is None else data_error

    # Fix data and data_error indexing if necessary
    if data_error is not None and data.shape != data_error.shape:
        raise ValueError("data and data error must have the same shape")
    if len(data.shape) == 1:
        data = _fix_nd_indexing(data)
        data_error = _fix_nd_indexing(data_error) if data_error is not None else None

    if len(iterations.ivars) != 2:
        raise ValueError(
            "This plot only works to plot data from an experiment with two independent variables")

    x_var, x_var_ind = get_var(iterations, x_var)
    it_var, it_var_ind = get_var(iterations, it_var)
    if it_var not in iterations.independent_variables:
        raise ValueError("iterated variable must be an independent variable")
    if it_var == x_var:
        raise ValueError("iterated variable must be distinct from x_var")

    no_guesses = None
    if guess is None:
        if not (guesses and g_split):
            raise ValueError("If guess is not provided BOTH guesses and g_split must be provided")
        no_guesses = len(guesses)
        if no_guesses == 0:
            raise ValueError("Guesses cannot be empty list. Provide valid guesses")
        elif no_guesses == 1:
            guess = guesses[0]
            no_guesses = None
            fit_dim = len(guess)
        elif no_guesses >= 2:
            try:
                tmp = g_split[0]  # check if g_split is iterable
            except TypeError:
                # if not set to a value in a list
                g_split = [g_split]
            dims = [len(g) for g in guesses]
            if not all([d == dims[0] for d in dims]):
                raise ValueError("All guesses must have the same dimensions")
            fit_dim = dims[0]
        if no_guesses is not None:
            try:
                # Make sure all values in g_split are usable as list indices
                g_split = [int(cut) for cut in g_split]
                g_split = g_split[:no_guesses - 1]
            except ValueError as ve:
                raise TypeError("All values in g_split must be ints") from ve
    else:
        fit_dim = len(guess)

    popts = zeros((len(iterations.independent_variables[it_var]), shots, fit_dim), dtype=float)
    perrs = zeros(popts.shape, dtype=float)

    for shot in range(shots):
        data_nd = iterations.fold_to_nd(data[:, shot])
        error_nd = iterations.fold_to_nd(data_error[:, shot]) if data_error is not None else [None]*len(data_nd)
        if it_var_ind:  # Transpose the data array if the index is right, for convenience
            #print("Transposing data_nd")
            data_nd = data_nd.T
            error_nd = error_nd.T if data_error is not None else error_nd

        # Add Redundant split to ensure for loop works nicely
        if no_guesses is not None:
            g_split.append(len(data_nd))

        for i, data_vals in enumerate(data_nd):
            if x_var in iterations.ivars:
                x_vals = sorted(iterations.independent_variables[x_var])  # values taken on the x-axis
            else:
                x_indep = iterations.ivars[not it_var_ind]
                x_len = len(iterations.independent_variables[x_indep])
                if it_var_ind:
                    x_vals = iterations[x_var][i*x_len:(i+1)*x_len]
                else:
                    x_vals = iterations[x_var][i::len(iterations.independent_variables[it_var])]
                #print(f"i, data_vals, len(data_vals) = {i}, {data_vals}, {len(data_vals)}")
                #print(x_indep, x_len, x_vals)
                #print(f"len(x_vals) = {len(x_vals)}")
            # use correct guess params if using guesses arg
            if no_guesses is not None:
                for j, cut in enumerate(g_split):
                    if i < cut:
                        break
                guess = guesses[j]
                print(f"Multiple guesses detected. Using Guess {j} = {guess}")
            data_ers = error_nd[i]
            # value of the y_ivar for this run through the loop
            y_val = sorted(iterations.independent_variables[it_var])[i]
            try:
                popt, pcov = curve_fit(func, x_vals, data_vals, sigma=data_ers, p0=guess)
            except RuntimeError as er:
                print(f"Failed to perform fit on dataset {i}: {er}")
                perrs[i, shot, :] = zeros(len(guess))
                popts[i, shot, :] = array([nan]*len(guess))
            else:
                perrs[i, shot, :] = sqrt(diag(pcov))
                popts[i, shot, :] = popt

    return popts, perrs


def fit_and_plot_hist(func, counter_data, bns, guess, title="", plots=True, **kwargs):
    hist, bin_edges = histogram(counter_data, bins=bns)
    mids = (bin_edges[0:-1] + bin_edges[1:]) / 2
    try:
        popt, pcov = curve_fit(f=func, xdata=mids, ydata=hist, p0=guess)
    except RuntimeError as e:
        print(f"Unable to fit\n{e}")
        popt = array(guess)
        pcov = zeros((len(popt), len(popt)))
        bad_fit = True
    else:
        bad_fit = False
    perr = sqrt(diag(pcov))

    if "figsize" in kwargs.keys():
        fs = kwargs["figsize"]
    else:
        fs = (5,5)
    if plots:
        fig, ax = plt.subplots(1, 1, figsize=fs)
        xlin = linspace(min(counter_data), max(counter_data), 1000)
        ax.hist(counter_data, bins=bns, histtype='step', label="Raw Data")
        label = "Guess" if bad_fit else "Fit"
        ax.plot(xlin, func(xlin, *popt), label=label)
        ax.set_title(title.format(popt, perr))
        ax.legend()
        fig.tight_layout()
        fig.show()
    else:
        fig, ax = (None, None)
    return popt, pcov, perr, fig, ax


# Functions useful for fitting  --------------------------------------------------------------------
# Related to retention scans  ----------------------------------------------------------------------
def exp_decay(x: float, a: float, Tau: float):
    """
    Exponential decay
    Args:
        x: time(s) to evaluate function
        a: magnitude of decay
        Tau: characteristic decay time

    Returns:
        a*exp(-x/Tau)
    """
    return a*exp(-x/Tau)


def o_m_exp_decay(x: float, a: float, Tau: float):
    """
    One 1 exp decay
    Args:
        x: time(s) to evaluate function
        a: magnitude of function
        Tau: characteristic decay time

    Returns:
        a*(1-exp_decay(x, 1, Tau))
    """
    return a*(1-exp_decay(x, 1, Tau))


def rabi_cos(x: float, freq: float, amp: float, offset: float, phi: float = pi):
    """
    Computes the probability of being in one of two states due to a Rabi Oscillation
    Args:
        x: time(s) to evaluate function
        freq: rabi frequency of oscillation
        amp: amplitude of oscillation
        offset: oscillation of oscialltion from 0
        phi: phase of oscillation, pi if starting in other state
    Returns:
        offset + amp*cos(freq * x / 2 + phi)**2
    """
    return offset + amp*cos(freq * x / 2 + phi)**2


def lorenz(x: float, x0: float, fwhm: float, a: float, o: float) -> float:
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


# Related to photon counting  ----------------------------------------------------------------------
def gaussian(x: Union[float, ndarray], mu: float, std: float, a: float) -> Union[float, ndarray]:
    """
    Samples a gaussian distribution at position(s) x

    Distribution is a * exp(((x - mu) / (sqrt(2) * std)) ** 2) / (std * sqrt(2 * pi))
    Args:
        x : position(s) to sample the function
        mu : mean/center of the distribution
        std : standard deviation of the distribution
        a : amplitude of the distribution. If a == 1 the distribution is a normalized probability
            distribution

    Returns:
        value(s) of the gaussian distribution at x
    """

    return a * exp(-((x - mu) / (sqrt(2) * std)) ** 2) / (std * sqrt(2 * pi))


def double_gaussian(x: Union[float, ndarray],
                    mu1: float,
                    mu2: float,
                    std1: float,
                    std2: float,
                    a1: float,
                    a2: float) -> Union[float,ndarray]:
    """
    Samples a distribution made up of two gaussians at position(s) x

    Distribution is gaussian(x, mu1, std1, a1)+gaussian(x, mu2, std2, a2)
    Args:
        x : position(s) to sample the function
        mu1 : mean/center of the first distribution
        mu2 : mean/center of the second distribution
        std1 : standard deviation of the first distribution
        std2 : standard deviation of the second distribution
        a1 : amplitude of the first distribution. If a1 == 1 the distribution is a normalized
            probability distribution
        a2 : amplitude of the second distribution. If a1 == 1 the distribution is a normalized
            probability distribution

    Returns:
        value(s) of the double gaussian distribution at x
    """
    return gaussian(x, mu1, std1, a1)+gaussian(x, mu2, std2, a2)


def poisson(x: Union[float, ndarray], mu: float, a: float) -> Union[float,ndarray]:
    """
    Samples a continuous poisson distribution at position(s) x

    Due to the use of gamma functions this function usually breaks down at high mu and x
    Args:
        x : position(s) to sample the function
        mu : mean/center of the distribution
        a : amplitude of the first distribution

    Returns:
        value(s) of the poisson distribution at x
    """
    return a*exp(-mu)*mu**x/gamma(x+1)


def double_poisson(x: Union[float, ndarray], mu1: float, mu2: float, a1: float, a2: float) -> \
        Union[float, ndarray]:
    """
    Samples a distribution made up of two poissonians at position(s) x

    Due to the use of gamma functions this function usually breaks down at high mu and x
    Args:
        x : position(s) to sample the function
        mu1 : mean/center of the first distribution
        mu2 : mean/center of the second distribution
        a1 : amplitude of the first distribution
        a2 : amplitude of the second distribution

    Returns:
        value(s) of the double poissonian distribution at x
    """
    return poisson(x, mu1, a1) + poisson(x, mu2, a2)


# Functions to find cuts, and understand errors due to overlaps, and choices in cut ----------------
def gauss_intercepts(
        mu1: float,
        mu2: float,
        std1: float,
        std2: float,
        a1: float,
        a2: float) -> Tuple[float, float, float, float, float]:
    """
    Intercepts between two gaussian distributions

    This function returns the numerical solution to the equation:
    > gauss(x,mu1,std1,a1) = gauss(x,mu2,std2,a2)
    for x.

    That solution is found analytically, with the solution in terms of the quadratic formula:
    (-b + sgn * sqrt( b ** 2 - 4 * a * c )) / (2 * a)
    where sgn takes the values +1 or -1, and 'a', 'b', and 'c' are computed explicitly.

    This function returns both intercepts as well as the 'a', 'b', and 'c' parameters of the
    quadratic formula above.

    Args:
        mu1 : mean/center of the first distribution
        mu2 : mean/center of the second distribution
        std1 : standard deviation of the first distribution
        std2 : standard deviation of the second distribution
        a1 : amplitude of the first distribution. If a1 == 1 the distribution is a normalized
            probability distribution
        a2 : amplitude of the second distribution. If a1 == 1 the distribution is a normalized
            probability distribution

    Returns:
        tuple(intecept1, intercept2, a, b, c)
            intercepts: value of the intercepts of the two distributions
            a: value of 'a' parameter of quadratic formula
            b: value of 'b' parameter of quadratic formula
            c: value of 'c' parameter of quadratic formula
    """
    a: float = 1 / std2 ** 2 - 1 / std1 ** 2
    b: float = -2 * (mu2 / std2 ** 2 - mu1 / std1 ** 2)
    c: float = (mu2 / std2) ** 2 - (mu1 / std1) ** 2 + 2 * (log(a1 / a2) - log(std1 / std2))

    intercepts = (
        (-b + sqrt(b ** 2 - 4 * a * c)) / (2 * a),  # plus case
        (-b - sqrt(b ** 2 - 4 * a * c)) / (2 * a)   # minus case
    )

    return intercepts[0], intercepts[1], a, b, c


def gauss_cut(
        mu1: float,
        mu2: float,
        std1: float,
        std2: float,
        a1: float,
        a2: float,
        intercepts: Tuple[float, float] = None) -> Tuple[float, int]:
    """
    Intercept between two gaussian distributions

    This value is also the optimal cut for minimizing discrimination errors.
    Args:
        mu1 : mean/center of the first distribution
        mu2 : mean/center of the second distribution
        std1 : standard deviation of the first distribution
        std2 : standard deviation of the second distribution
        a1 : amplitude of the first distribution. If a1 == 1 the distribution is a normalized
            probability distribution
        a2 : amplitude of the second distribution. If a1 == 1 the distribution is a normalized
            probability distribution
        intercepts : pre-computed intercepts

    Returns:
        cut : float, value of the intercept between the two distributions
            sgn : sign chosen in the quadratic formula

    Raises:
        ValueError: If neither of the two intercepts is inbetween mu1 or mu2
    """
    try:
        if intercepts is None:
            intercepts = gauss_intercepts(mu1, mu2, std1, std2, a1, a2)
        cut = list(filter(
            lambda x: mu1 < x < mu2,
            intercepts
        ))[0]
        sgn = 1 if cut == intercepts[0] else -1
    except IndexError:
        raise ValueError(
            "Neither intercept is within acceptable range, you have bigger problems than this error"
        )
    else:
        return cut, sgn


def gauss_discrimination_error(
        xc: float,
        mu1: float,
        mu2: float,
        std1: float,
        std2: float,
        a1: float,
        a2: float) -> float:
    """
    Returns the error rate discriminating between two gaussian curves given a cut xc.

    normalizes the distributions by setting a1 = a1 / (a1 + a2) and a2 = a2 / (a1 + a2)

    Only works if mu1 < mu2

    Args:
        xc : cut used to discriminate between the lower and higher gaussian curve
        mu1 : mean/center of the first distribution
        mu2 : mean/center of the second distribution
        std1 : standard deviation of the first distribution
        std2 : standard deviation of the second distribution
        a1 : amplitude of the first distribution. If a1 == 1 the distribution is a normalized
            probability distribution
        a2 : amplitude of the second distribution. If a2 == 1 the distribution is a normalized
            probability distribution

    Returns:
        expected fraction of events expected to result in an incorrect classification based on
            the cut

    Raises:
        ValueError: if mu1 > mu2 it raises a value error to ensure the user does not end up using
            bad results from a poor application of this function
    """
    if mu1 > mu2:
        raise ValueError("mu1 must be less than mu2!")
    a1 = a1 / (a1 + a2)
    a2 = a2 / (a1 + a2)
    return 0.5*(a1*(1-erf((xc-mu1) / (sqrt(2)*std1))) + a2*(1+erf((xc-mu2) / (sqrt(2)*std2))))


def gauss_overlap(
        mu1: float,
        mu2: float,
        std1: float,
        std2: float,
        a1: float,
        a2: float) -> float:
    """
    Returns the fraction of a double gaussian distribution that corresponds to the overlap between
        the two gaussian curves

    assumes mu1 < mu2

    Distribution is gaussian(x, mu1, std1, a1)+gaussian(x, mu2, std2, a2)
    Args:
        mu1 : mean/center of the first distribution
        mu2 : mean/center of the second distribution
        std1 : standard deviation of the first distribution
        std2 : standard deviation of the second distribution
        a1 : amplitude of the first distribution. If a1 == 1 the distribution is a normalized
            probability distribution
        a2 : amplitude of the second distribution. If a2 == 1 the distribution is a normalized
            probability distribution

    Returns:
        fractional error discriminating between two peaks given a cut
    """
    cut, sgn = gauss_cut(mu1, mu2, std1, std2, a1, a2)
    return gauss_discrimination_error(cut, mu1, mu2, std1, std2, a1, a2)


def poisson_intercept(mu1: float, mu2: float, a1: float, a2: float) -> \
        Union[float, ndarray]:
    """
    Returns the intercept between two poisson distributions. Assuming mu1 < mu2.

    This value is also the optimal cut for minimizing discrimination errors.
    Args:
        mu1 : mean/center of the first distribution
        mu2 : mean/center of the second distribution
        a1 : amplitude of the first distribution
        a2 : amplitude of the second distribution

    Returns:
        value of the intercept of the two distributions
    """
    return (log(a2/a1)-(mu2-mu1))/log(mu1/mu2)


def poisson_discrimination_error(xc: float, mu1: float, mu2: float, a1: float, a2: float) -> float:
    """
    Returns the error rate discriminating between two poisson curves given a cut xc.

    normalizes the distributions by setting a1 = a1 / (a1 + a2) and a2 = a2 / (a1 + a2)

    Only works if mu1 < mu2

    Args:
        xc : cut used to discriminate between the lower and higher poisson curve
        mu1 : mean/center of the first distribution
        mu2 : mean/center of the second distribution
        a1 : amplitude of the first distribution
        a2 : amplitude of the second distribution

    Returns:
        expected fraction of events expected to result in an incorrect classification based on
            the cut

    Raises:
        ValueError: if mu1 > mu2 it raises a value error to ensure the user does not end up using
            bad results from a poor application of this function
    """
    if mu1 > mu2:
        raise ValueError("mu1 must be less than mu2!")
    a1 = a1 / (a1 + a2)
    a2 = a2 / (a1 + a2)

    err_low_curve = integrate.quad(lambda x: poisson(x, mu1, a1), xc, mu1+10*sqrt(mu1))
    err_high_curve = integrate.quad(lambda x: poisson(x, mu2, a2), 0.0001, xc)
    return err_low_curve[0] + err_high_curve[0]


def poisson_overlap(mu1: float, mu2: float, a1: float, a2: float) -> float:
    """
    Returns the fraction of a double poisson distribution which is made up of the overlap between
        the two curves.

    Due to the use of gamma functions this function usually breaks down at high mu and x
    Args:
        mu1 : mean/center of the first distribution
        mu2 : mean/center of the second distribution
        a1 : amplitude of the first distribution
        a2 : amplitude of the second distribution

    Returns:
        value(s) of the double poissonian distribution at x
    """
    intercept = poisson_intercept(mu1, mu2, a1, a2)
    return poisson_discrimination_error(intercept, mu1, mu2, a1, a2)


# Uncertainties on derived values ------------------------------------------------------------------
def cut_err_poisson(mu1: float,
                    mu2: float,
                    a1: float,
                    a2: float,
                    dmu1: float = 0,
                    dmu2: float = 0,
                    da1: float = 0,
                    da2: float = 0
                    ) -> float:
    """
    Uncertainty in the cut value derived for discriminating between two poisson distributions

    Specifics on this computation are contained in
        H5_python3/Documentation/Gauss_and_Poisson_fit_functions.pdf

    Args:
        mu1 : mean/center of the first distribution
        mu2 : mean/center of the second distribution
        a1 : amplitude of the first distribution
        a2 : amplitude of the second distribution
        dmu1 : uncertainty in mean/center of the first distribution
        dmu2 : uncertainty in mean/center of the second distribution
        da1 : uncertainty in amplitude of the first distribution
        da2 : uncertainty in amplitude of the second distribution

    Returns:
        Uncertainty in the cut value
    """
    inv_lg = 1/log(mu1/mu2)  # convenient value to compute only once
    return inv_lg*(da2/a2-da1/a1+(dmu1-dmu2)*(1-inv_lg))


def cut_err_gauss(
        mu1: float,
        mu2: float,
        std1: float,
        std2: float,
        a1: float,
        a2: float,
        dmu1: float = 0,
        dmu2: float = 0,
        dstd1: float = 0,
        dstd2: float = 0,
        da1: float = 0,
        da2: float = 0
) -> float:
    """
    Uncertainty in the cut value derived for discriminating between two gaussian distributions

    Specifics on this computation are contained in
        H5_python3/Documentation/Gauss_and_Poisson_fit_functions.pdf

    Args:
        mu1 : mean/center of the first distribution
        mu2 : mean/center of the second distribution
        std1 : standard deviation of the first distribution
        std2 : standard deviation of the second distribution
        a1 : amplitude of the first distribution. If a1 == 1 the distribution is a normalized
            probability distribution
        a2 : amplitude of the second distribution. If a2 == 1 the distribution is a normalized
            probability distribution
        dmu1 : uncertainty in mean/center of the first distribution
        dmu2 : uncertainty in mean/center of the second distribution
        dstd1 : uncertainty in standard deviation of the first distribution
        dstd2 : uncertainty in standard deviation of the second distribution
        da1 : uncertainty in amplitude of the first distribution. If a1 == 1 the distribution is a
            normalized probability distribution
        da2 : uncertainty in amplitude of the second distribution. If a2 == 1 the distribution is a
            normalized probability distribution

    Returns:
        uncertainty in the cut value
    """

    intercept_p, intercept_m, a, b, c = gauss_intercepts(mu1, mu2, std1, std2, a1, a2)

    cut, sgn = gauss_cut(mu1, mu2, std1, std2, a1, a2, intercepts=(intercept_p, intercept_m))

    # uncertainty in derived a, b, c parameters we feed to the quadratic formula
    da = 2 * (dstd1 / std1 ** 3 - dstd2 / std2 ** 3)
    db = -2 * (
                -dmu1 / std1 ** 2 + dmu2 / std2 ** 2 + 2 * dstd1 * mu1 / std1 ** 3 - 2 * dstd2 * mu2 / std2 ** 3)
    dc = 2 * (-dmu1 * mu1 / std1 ** 2 + dmu2 * mu2 / std2 ** 2 + dstd1 * (
                mu1 ** 2 / std1 ** 3 - 1 / std1 ** 2) + dstd2 * (
                      -mu2 ** 2 / std2 ** 3 + 1 / std2 ** 2) + da1 / a1 - da2 / a2)

    inva = 1 / a
    q = sqrt(b ** 2 - 4 * a * c)  # discriminant

    # cut uncertainty
    dcut = inva * (da * (-(-b + sgn * q) / a - sgn * 2 * c / q) + db * (-1 + sgn * b / q) - sgn * dc * 2 * a / q)

    return abs(dcut)



