import math
from numpy import *
from typing import Union, Tuple
from scipy.special import gamma, erf
from scipy import integrate


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


# Functions useful for fitting  --------------------------------------------------------------------
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

    err_low_curve = integrate.quad(lambda x: poisson(x, mu1, a1), xc, inf)
    err_high_curve = integrate.quad(lambda x: poisson(x, mu2, a2), 0, xc)
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
