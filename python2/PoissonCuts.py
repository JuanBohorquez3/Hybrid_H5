from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import gamma, gammaincc, factorial
from scipy.optimize import curve_fit

def double_poisson(x, rate1, rate2, frac1, frac2):
    """
    Generates a two poisson curve along x
    :param x: position(s) to plot curve
    :param rate1: mean scattering rate of the lower curve
    :param rate2: mean scattering rate of the higher curve
    :param frac1: fraction of events which occur with the first rate
    :param frac2: fraction of events which occur with the second rate
    :return: probability of detecting x events in a two poisson process
    """
    if rate1 < 80:
        p1 = frac1*np.exp(-rate1)*rate1**(x)/factorial(x)
    else:
        std = np.sqrt(np.abs(rate1))
        #print rate2
        #print std
        z = x-rate1
        p1 = frac1*np.exp(-z**2/(2*std**2))/np.sqrt(2*np.pi*std**2)

    if rate2 < 80:
        p2 = frac2*np.exp(-rate2)*rate2**(x)/factorial(x)
    else:
        std = np.sqrt(np.abs(rate2))
        #print rate2
        #print std
        z = x-rate2
        p2 = frac2*np.exp(-z**2/(2*std**2))/np.sqrt(2*np.pi*std**2)

    return p1+p2


def cut_err(rlow, rhigh, nc):
    """
    returns an average error rate for discriminating between poisson processes with rates rlow and rhigh given a cutoff
    rate nc
    :param rlow: low event rate
    :param rhigh: high event rate
    :param nc: cutoff count rate
    :return: average error rate for a 50/50 split between processes
    """
    return 1-gammaincc(nc, rlow)+gammaincc(nc, rhigh)


def best_cut(rlow, rhigh):
    """
    use brute force to find the cut that maximizes discrimination between two poisson processes with rates rlow and
    rhigh
    :param rlow: low event rate
    :param rhigh: high event rate
    :return: nc, cutoff rate which minimizes false positive and false negative rates
    """
    rlow = int(rlow)
    rhigh = int(rhigh)
    print "rlow= {}; rhigh = {}".format(rlow,rhigh)
    n = np.arange(rlow, rhigh+1, 1.0)
    print n
    er = cut_err(rlow, rhigh, n)
    ind = np.where(er == min(er))
    if len(ind[0]) > 1:
        ind = ind[0][len(ind)+1]
    return n[ind]
    return n[ind]

def poisson_fit(data, m0=None, m1=None, f0=None, f1=None):
    """
    fits data to a double poissonian curve
    :param data: data to be fit
    :param m0: guess for the low rate
    :param m1: guess for the high rate
    :param f0: guess for the low rate probability
    :param f1: guess for the high rate probability
    :return: params: best fit parameters for the data to the double poissonian
             perr: error bars for the above parameters
             cut: the optimal cutoff rate which minimizes discrimination error
    """
    guess = [m0,m1,f0,f1]
    bn = 3*np.array(range(int(max(data)/3)))
    h = np.histogram(data,bins=bn)
    #notmalize the histogram to ease the fitting
    y = h[0]/float(h[0].sum())
    #find the fit parameters
    try:
        popt, pcov = curve_fit(double_poisson,h[1][1:],y,guess)
        params = abs(popt)
        perr = np.sqrt(abs(np.diag(pcov)))
    except RuntimeError as e:
        popt = None
        pcov = None
        params = None
        perr = None
        print(e)
    #keep the order of params right
    if params is not None:
        if(params[0] > params[1]):
            bf = params[0,2]
            params[0,2] = params[1,3]
            params[1,3] = bf
            bf = perr[0,2]
            perr[0,2] = perr[1,3]
            perr[1,3] = bf

    #get an optimal cut
    if params is not None:
        cut = best_cut(params[0],params[1])
    else :
        cut = 0

    return params, perr, cut

def poisson_cuts(data,
                 mode = 'fit',
                 label = None,
                 force_cuts=False,
                 retention_data=None,
                 Bg_data=None,
                 m0=None, m1=None, f0=None, f1=None):
    """
    A function that fits data to a two poissonian curve then extracts the optimal cut for discriminating between the two
    poisson processes in question
    :param data: ndarray, initial counter data used to determine fits
    :param mode: string, mode of use for this function.
        Modes: 'fit' provides a fit and cut to data, and determines loading fractions
               'retention' provides a fit and cut to data, then uses the fit and cut to determine the retention rate
    :param label: string, a label for the plots to be plotted
    :param force_cuts: boolean, should the cut from the data fit be used in the retention fit?
    :param retention_data: ndarray, retention counter data
    :param Bg_data: ndarry, a data set where only one of the two poisson processes is present
    :param m0: float, a guess for the low rate
    :param m1: float, a guess for the high rate
    :param f0: float, a guess for the fraction of low events
    :param f1: float, a guess for the fraction of high events
    :return: params: ndarray, the fit parameters for the double poissonian
             perr: ndarrya, the error bars for params
             cut: float, the optimal cut between the processes
             rload: float, the fraction of evens with the high rate, based on cut
             retention: float, the retention rate. Only returned in 'retention' mode
    """

    guess = np.zeros(4,dtype=float)

    err_missing = 'Error: {} is missing'
    err_type = 'Error: {} should be of type {}'
    err_shape = 'Error: {} (shape = {}) should be the same shape as {} (shape = {})'
    assert isinstance(data, type(np.zeros(1))), err_type.format("data",type(np.zeros(1)))
    if mode == 'retention':
        assert retention_data is not None, err_missing.format('retention_data')
        assert isinstance(retention_data, type(np.zeros(1))), err_type.format('retention_data',
                                                                                type(np.zeros(1)))
        assert (retention_data.shape == data.shape), err_shape.format('retention_data',
                                                                      retention_data.shape,
                                                                      'data',
                                                                      data.shape)
    if Bg_data is not None:
        assert isinstance(Bg_data, type(np.zeros(1))), err_type.format("Bg_data", type(np.zeros(1)))
        m0 = Bg_data.mean()

    if m0 != None:
        guess[0] = m0
    if m1 != None:
        guess[1] = m1
    if f0 != None:
        guess[2] = f0
    if f1 != None:
        guess[3] = f1

    params, perr, cut = poisson_fit(data,*guess)

    #determine the loading fraction
    if params is not None:
        loaded = np.where(data>cut)
        load_data = data[loaded]
        rload = len(load_data)/len(data)
    else:
        loaded = 0
        load_data = 0
        rload = -1

    if mode == 'fit':
        bn = 2 * np.array(range(int(max(data) / 2)))

        #dbg
        print('fitting')
        fig, ax = plt.subplots(1,1,figsize=(6,6))
        ax.hist(data,bins=bn,normed=True)
        x_dat = np.linspace(0,max(data),2000)
        if params is not None:
            ax.plot(x_dat,double_poisson(x_dat,*params))
            ax.plot([cut]*2,ax.get_ylim(),'k')
            ax.set_title("{0}\n m0 = {1[0]}, m1 = {1[1]}\nf0 = {1[2]}, f1 = {1[3]}\ncut={2}".format(label,params,cut))
        fig.tight_layout()
        plt.show()
        return params, perr, cut, rload
    if mode == 'retention':
        #fit retention data to another double poissonian using previously derived parameters as a guess
        rparams, rperr, rcut = poisson_fit(retention_data,*params)
        if force_cuts:
            rcut = cut

        #dbg
        print('retentioning')


        retained = np.where(retention_data>rcut)
        retained_data = retention_data[retained]
        retention = len(retained_data)/len(load_data)
        fig, axarr = plt.subplots(1,3,figsize=(12,6))

        bn = range(int(max(data)))

        #plot the data fit
        axarr[0].hist(data,bins=bn,normed=True)
        x_dat = np.linspace(0,max(data),2000)
        axarr[0].plot(x_dat,double_poisson(x_dat,*params))
        axarr[0].plot([cut]*2,axarr[0].get_ylim(),'k')
        axarr[0].set_title("m0 = {}, m1 = {},\n f0 = {}, f1 = {}".format(*params))

        #plot the retention data fit
        axarr[1].hist(retention_data,bins=bn,normed=True)
        x_dat = np.linspace(0,max(data),2000)
        axarr[1].plot(x_dat,double_poisson(x_dat,*rparams))
        axarr[1].plot([rcut]*2,axarr[0].get_ylim(),'k')
        axarr[1].set_title("m0 = {}, m1 = {},\n f0 = {}, f1 = {}".format(*rparams))

        #plot the retention_data for measurements that were loaded in the data shot
        axarr[2].hist(retention_data[loaded],bins=bn,normed=True)
        axarr[2].plot([rcut] * 2,axarr[0].get_ylim(),'k')
        axarr[2].set_title("Retention : {}".format(retention))
        plt.suptitle("{}".format(label))

        return params, perr, cut, retention