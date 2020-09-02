__author__ = 'Hybrid'

'''

Image load and analysis script based on
counterh5.py by Josh Isaacs
and Gauss_fit.ipynb by Don Booth

2016/06/14

Examples of usage (in Jupyter notebook):
    To load a time-of-flight data set and measure the temperature of a MOT:
        imageH5TOF = i5.ImageH5("Z:\\Public\\Hybrid\\Data\\Measurements\\2016_06_14\\2016_06_14_09_39_03_TOF_TEST")     #load data
        axTOF = imageH5TOF.showGraphNoContours(0,0)                                                                     #show graph of first iteration
            (use graph to select ROI)
        imageH5TOF.getLimits(axTOF)                                                                                     #get ROI limits from graph
        imageH5TOF.LoadFit()                                                                                            #fit images to gaussians
        imageH5TOF.getTemperaturesXY()                                                                                  #fit gaussian params to get temperatures in X/Y axes

    To load absorption images and display them:
        imageH5 = i5.ImageH5("D:\\Hybrid Experiment\\Data\\2016_06_15\\2016_06_15_14_15_53_TransportXScan", '329', 2)   #load data for camera serial 329, position 2 in Python controller
        imageH5.useAbsorptionImages()                                                                                   #Tells imageH5 to expect two shots, one BG one signal
        ax = imageH5.showGraphNoContours(15,10)                                                                         #shows graph of iteration 15, measurement 10

    To load lifetime data from a trap where the signal is too small to fit to Gaussians:
        imageH5trap = i5.ImageH5("Z:\\Public\\Hybrid\\Data\\Measurements\\2016_06_15\\2016_06_15_09_57_00_PythonTrap_Lifetime") #load data
        axtrap = imageH5trap.showGraphNoContours(0,0)                                                                   #show graph of first iteration
            (use graph to select a narrow ROI)
        imageH5trap.getLimits(axtrap)                                                                                   #get ROI limits from graph
        imageH5trap.sumOverRegion()                                                                                     #integrate over ROI for number of counts
        imageH5trap.LifetimeCurve()                                                                                     #fit counts to decaying exponential for lifetime
'''

import h5py
from numpy import *
import os
import matplotlib.pyplot as plt
import scipy.optimize as opt
import counterh5 as ch5
import seaborn as sbs
import itertools
import zipfile


def seedpermute(arr, seed):
    random.seed(seed)
    return random.permutation(arr)



class ImageH5():
    
    def __init__(self,directory,cameraID='1026',camnum=0,expHDF=None):
        if expHDF is None:
            os.chdir(directory)
            self.resultsfile = self.openfile("results.hdf5")
        else:
            self.resultsfile = expHDF
        self.cameraID = cameraID
        self.camnum = camnum
        self.minIter=0
        self.maxIter=self.getlen('iterations/')
        self.minMeas=0
        self.maxMeas = min([self.getlen('iterations/{}/measurements/'.format(i)) for i in range(self.maxIter+1)])
        self.allandorimages = []
        self.binMode = self.getBinMode()
        self.allpopts = []
        self.xT = 0
        self.yT = 0
        self.lifetime = 0
        self.poptmethod = ''
        self.xlimit = []
        self.ylimit = []
        self.imagefunction = self.imagefromh5
        self.iterations = len(self.resultsfile['experiments/0/iterations/'].items())
        self.iVars=[]
        if self.iterations > 1:
            for i in self.resultsfile['settings/experiment/independentVariables/'].iteritems(): 
                tmp=eval(i[1]['function'][()])
                if (type(tmp) == list) | (type(tmp) == ndarray) | (type(tmp) == tuple): self.iVars.append((i[0],i[1]['function'][()],i[1]['description'][()]))
        self.iVarVals = list(itertools.product(*[eval(self.iVars[i][1]) for i in range(len(self.iVars))]))
    
    def openfile (self, filename):
        if not os.path.isfile(filename):
            filename = filename+".zip"
            if not os.path.isfile(filename):
                raise "File not found. Does this directory contain a results.hdf5 or results.hdf5.zip?"
        if zipfile.is_zipfile(filename):
            zf = zipfile.ZipFile(filename)
            filecontents = zf.read(os.path.basename(os.path.splitext(filename)[0]))
            tempfile = open("D:\\unzipped.hdf5","wb")
            tempfile.write(filecontents)
            tempfile.close()
            filename = "D:\\unzipped.hdf5"
        return h5py.File(filename)
    
    # -------------------  Fitting Functions

    # twoD_Gaussian: 2D Gaussian function with variable theta for major/minor axis. Uses sigma instead of W for width.
    def twoD_Gaussian(self, (x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        xo = float(xo)
        yo = float(yo)
        a = (cos(theta)**2)/(2*sigma_x**2) + (sin(theta)**2)/(2*sigma_y**2)
        b = -(sin(2*theta))/(4*sigma_x**2) + (sin(2*theta))/(4*sigma_y**2)
        c = (sin(theta)**2)/(2*sigma_x**2) + (cos(theta)**2)/(2*sigma_y**2)
        g = offset + amplitude*exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                                + c*((y-yo)**2)))
        return g.ravel()

    def twoD_Gaussian_W1(self, (x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset0, offset1):
        xo = float(xo)
        yo = float(yo)
        a = (cos(theta)**2)/(2*sigma_x**2) + (sin(theta)**2)/(2*sigma_y**2)
        b = -(sin(2*theta))/(4*sigma_x**2) + (sin(2*theta))/(4*sigma_y**2)
        c = (sin(theta)**2)/(2*sigma_x**2) + (cos(theta)**2)/(2*sigma_y**2)
        g = offset0 + offset1*x + amplitude*exp( - (4*a*((x-xo)**2) + 8*b*(x-xo)*(y-yo)
                                + 4*c*((y-yo)**2)))
        return g.ravel()
    
    # oneD_Gaussian: 1D Gaussian function. Uses sigma instead of W for width.
    def oneD_Gaussian(self, x, amplitude, xo, sigma_x, offset):
        xo = float(xo)
        a = (1.0)/(2*sigma_x**2)
        g = offset + amplitude*exp(-a*((x-xo)**2))
        return g
                                   
    # oneD_Gaussian_W: 1D Gaussian function. Uses W instead of sigma for width.
    def oneD_Gaussian_W(self, x, amplitude, xo, W0_x, offset):
        xo = float(xo)
        a = (2.0)/(W0_x**2)
        g = offset + amplitude*exp(-a*((x-xo)**2))
        return g
    # oneD_Gaussian_W: 1D Gaussian function with 1st Order Offset. Uses W instead of sigma for width.
    def oneD_Gaussian_W1(self, x, amplitude, xo, W0_x, offset0, offset1):
        xo = float(xo)
        a = (2.0)/(W0_x**2)
        g = offset0 + offset1*x + amplitude*exp(-a*((x-xo)**2))
        return g
    
    # twoD_Gaussian_NoAngle: 2D Gaussian function which assumes major/minor axis are along x/y. Uses sigma instead of W for width.
    def twoD_Gaussian_NoAngle(self, (x, y), amplitude, xo, yo, sigma_x, sigma_y, offset):
        xo = float(xo)
        yo = float(yo)
        a = 1.0/(2*sigma_x**2)
        b = 0
        c = 1.0/(2*sigma_y**2)
        g = offset + amplitude*exp( - (a*((x-xo)**2) + c*((y-yo)**2)))
        return g.ravel()

    # twoD_Gaussian_NoAngle_W: 2D Gaussian function which assumes major/minor axis are along x/y. Uses W for width.
    def twoD_Gaussian_NoAngle_W(self, (x, y), amplitude, xo, yo, W0_x, W0_y, offset):
        xo = float(xo)
        yo = float(yo)
        a = 2.0/(W0_x**2)
        b = 0
        c = 2.0/(W0_y**2)
        g = offset + amplitude*exp( - (a*((x-xo)**2) + c*((y-yo)**2)))
        return g.ravel()

    # LineFunction: Linear function.
    def LineFunction(self, x, slope, intercept):
        g = intercept + slope*x
        return g

    # QuadraticFunction: Quadratic function a*x**2 + b*x + c
    def QuadraticFunction(self, x, a, b, c):
        g = a*x*x + b*x + c
        return g

    # QuadraticFunctionNoB: Quadratic function with no linear term a*x**2 + c
    def QuadraticFunctionNoB(self, x, a, c):
        g = a*x*x + c
        return g

    # SqrtQuadraticFunctionNoB: Square Root of a Quadratic function with No Linear term: Sqrt(a*x**2 + c)
    def SqrtQuadraticFunctionNoB(self, x, a, c):
        g = sqrt(a*x*x + c)
        return g

    # DecayingExponential: Decaying exponential function, useful for fitting lifetimes
    def DecayingExponential(self, x, a, tau, b):
        g = a * exp(-x/tau) + b
        return g

    # -------------------- Image loading

    # getlen: Takes in an opened h5py File and a key, and returns the length of the key (minus one, what with zero indexing...)
    def getlen(self, key):
        return len(self.resultsfile[key].items())-1

    # loadImageStack: pass to it an opened h5py File and it returns the images from the Andor camera identified by cameraID.
    #                cameraID is optional and defaults to 1026.
    def loadImageStack(self):
        firstiter = self.resultsfile['iterations/{}/measurements/{}/data/Andor_{}'.format(self.minIter,self.minMeas,self.cameraID)][0]
        allandorimages = zeros((self.maxIter - self.minIter + 1,self.maxMeas - self.minMeas + 1,len(firstiter),len(firstiter[0])))
        for iteration in range(self.minIter,self.maxIter + 1,1):
            for measurement in range(self.minMeas,self.maxMeas + 1,1):
                allandorimages[iteration,measurement] = \
                self.resultsfile['iterations/{}/measurements/{}/data/Andor_{}'.format(iteration,measurement,self.cameraID)][0]
        self.allandorimages = allandorimages
        return allandorimages

    # -------------------- Graphing functions

    def showGraphNoContours(self, iteration,measurement=-1,shot=0,showit=True,title='',save=False,savefile='',rang=-1,subset=-1,interp='nearest'):
        try:
            if len(subset) != 2:
                subset = [[0,None],[0,None]]
        except:
            subset = [[0,None],[0,None]]
        subset = array(subset)
        
        if measurement != -1:
            andorimage = self.imagefunction(iteration,measurement,shot=shot)[subset[0,0]:subset[0,1],subset[1,0]:subset[1,1]]
        else:
            for m in range(self.minMeas,self.maxMeas+1,1):
                andorimage = mean(
                    array(
                        [self.imagefunction(iteration, m, shot=shot)[subset[0, 0]:subset[0, 1], subset[1, 0]:subset[1, 1]] for m in range(self.minMeas, self.maxMeas+1, 1)]
                    ), axis=0
                )
        print "max: {}, mean: {}".format(andorimage.max(),andorimage.mean())

        fig, ax = plt.subplots(1)
        
        #try:
        #    if len(subset) == 2:
        #        andorimage = andorimage[subset[0,0]:subset[0,1],subset[1,0]:subset[1,1]]
        #except:
        #    print ""
        
        if rang==-1 or len(rang) != 2:
            ax.imshow(andorimage, cmap='jet', interpolation=interp)
        else:
            ax.imshow(andorimage, cmap='jet', vmin=rang[0], vmax=rang[1], interpolation=interp)
        plt.title(title)

        if showit:
            fig.show()
        if save and savefile != '':
            fig.savefig(savefile)
        return ax



    # ShowGraph: Displays a graph with the image passed to it in andorimage, and uses the parameters passed in popt
    #           with the function passed in func (which defaults to twoD_Gaussian) to display contours of the fitted
    #           function.
    def showGraph(self, andorimage,popt,func=twoD_Gaussian):
        x = linspace(0,len(andorimage[0]) - 1,len(andorimage[0]))
        y = linspace(0,len(andorimage) - 1,len(andorimage))
        x,y = meshgrid(x,y)
        data_fitted = func((x,y),*popt)

        fig,ax = plt.subplots(1,1)
        ax.hold(True)
        ax.imshow(andorimage.reshape(len(andorimage),len(andorimage[0])),cmap='jet',origin='bottom',
                  extent=(x.min(),x.max(),y.min(),y.max()))
        ax.contour(x,y,data_fitted.reshape(len(andorimage),len(andorimage[0])),8,colors='w')
        fig.show()
    
    
    def showGraphBackSub(self,iteration,measurement=-1,showit=True,title='',save=False,savefile='', binMode=1,rang=-1,binme=False):
        if measurement != -1:
            andorimage = self.imagefunction(iteration,measurement,1) - self.imagefunction(iteration,measurement,0)
        else:
            if (binme):
                andorimage = mean(array([self.binimg(self.imagefunction(iteration,m,1)) - 
                                         self.binimg(self.imagefunction(iteration,m,0)) 
                                         for m in range(self.minMeas,self.maxMeas+1,1)]),axis=0)
            else:
                andorimage = mean(array([array(self.imagefunction(iteration,m,1)) - 
                                         array(self.imagefunction(iteration,m,0)) 
                                         for m in range(self.minMeas,self.maxMeas+1,1)]),axis=0)
        print "max: {}, mean: {}".format(andorimage.max(),andorimage.mean())

        fig, ax = plt.subplots(1)
        if rang==-1 or len(rang) != 2:  
            ax.imshow(andorimage, cmap='jet')
        else:
            ax.imshow(andorimage, cmap='jet', vmin=rang[0], vmax=rang[1])
        plt.title(title)

        if showit:
            fig.show()
        if save and savefile != '':
            fig.savefig(savefile)
        return ax

    def getLimits(self, ax):
        xlimit = array(ax.get_ylim(), dtype = int)
        ylimit = array(ax.get_xlim(), dtype = int)

        xlimit.sort()
        ylimit.sort()
        print 'ROI limits are\n x:{},{}\n y:{},{}\n'.format(xlimit[0],xlimit[1],ylimit[0],ylimit[1])
        self.xlimit = xlimit
        self.ylimit = ylimit
        return xlimit, ylimit


    # --------------------- Functions for fitting all iterations/measurements

    # fit_2DGauss_iterations: Fits all measurements for all iterations individually to a 2D Gaussian fn.
    #     Parameters:
    #            images: array of images to fit. Shape should be (iterations, measurements, camera-rows, camera-columns)
    #            miniter, maxiter: minimum and maximum value for iterations to be included
    #            minmeas, maxmeas: minimum and maximum value for measurements to be included
    #            xlimit: Two-element array which indicates min/max columns to include while fitting
    #            ylimit: Two-element array which indicates min/max rows to include while fitting
    #            showGraphs: Boolean which tells whether to show the graphs for each fit
    #     Returns: popt_array: Array of parameters for each fit. Shape: (maxiter - miniter + 1,maxmeas - minmeas + 1,7)
    def fit_2DGauss_iterations(self, images,miniter,maxiter,minmeas,maxmeas,xlimit,ylimit,showGraphs=False):
        popt_array = zeros((maxiter - miniter + 1,maxmeas - minmeas + 1,7))
        print popt_array.shape
        func = self.twoD_Gaussian_NoAngle_W
        for iteration in range(miniter,maxiter + 1,1):
            for measurement in range(minmeas,maxmeas + 1,1):
                andorimage = images[iteration,measurement]
                andorimage = andorimage[xlimit[0]:xlimit[1],ylimit[0]:ylimit[1]]
                x = linspace(0,len(andorimage[0]) - 1,len(andorimage[0]))
                y = linspace(0,len(andorimage) - 1,len(andorimage))
                x,y = meshgrid(x,y)

                initial_guess = (max(andorimage) - mean(andorimage),x.mean(),y.mean(),100,100,1000)
                print "Iteration: {}, Measurement: {}".format(iteration,measurement)
                try:
                    popt,pcov = opt.curve_fit(func,(x,y),andorimage.ravel(),p0=initial_guess)
                except Exception as e:
                    print "Fit failed: {}".format(e)
                    popt = (NaN,) * 7
                data_fitted = func((x,y),*popt)
                print popt
                for i,el in enumerate(popt):
                    popt_array[iteration - miniter,measurement - minmeas,i] = el

                try:
                    self.showGraph(andorimage,popt,func)
                except Exception as e:
                    print "Graph failed. Did fit fail? Exception: {}".format(e)
        self.allpopts = popt_array
        self.poptmethod = 'fit'
        return popt_array


    # fit_2DGauss_iterations_avg: Fits an average of all measurements for each iteration to a 2D Gaussian fn, using W instead of sigma.
    #     Parameters:
    #            images: array of images to fit. Shape should be (iterations, measurements, camera-rows, camera-columns)
    #            miniter, maxiter: minimum and maximum value for iterations to be included
    #            minmeas, maxmeas: minimum and maximum value for measurements to be included
    #            xlimit: Two-element array which indicates min/max columns to include while fitting
    #            ylimit: Two-element array which indicates min/max rows to include while fitting
    #            showGraphs: Boolean which tells whether to show the graphs for each fit
    #     Returns: popt_array: Array of parameters for each fit. Shape: (maxiter - miniter + 1,maxmeas - minmeas + 1,7)
    #     Example of usage:    allpopts = fit_2DGauss_iterations_avg(
    #                                     allandorimages,
    #                                     0,
    #                                     getlen(resultsfile,'iterations/'),      #number of iterations - 1
    #                                     0,
    #                                     getlen(resultsfile,'iterations/0/measurements/'),    #number of measurements - 1
    #                                     xlimit,
    #                                     ylimit,
    #                                     True)
    #
    #
    def fit_2DGauss_iterations_avg(self,miniter=0,maxiter=-1,minmeas=0,maxmeas=-1,showGraphs=False):
        xlimit = self.xlimit
        ylimit = self.ylimit
        if (maxiter == -1):
            maxiter = self.maxIter
        if (maxmeas == -1):
            maxmeas = self.maxMeas
        popt_array = zeros((maxiter-miniter+1,7))
        print popt_array.shape
        func = self.twoD_Gaussian_NoAngle_W
        for iteration in range(miniter,maxiter+1,1):
            andorimagemeas = zeros((maxmeas-minmeas+1,xlimit[1]-xlimit[0],ylimit[1]-ylimit[0]))
            for measurement in range(minmeas,maxmeas+1,1):
                andorimagemeas[measurement] = self.imagefunction(iteration,measurement)[xlimit[0]:xlimit[1],ylimit[0]:ylimit[1]]
            andorimage = andorimagemeas.mean(0)
            print andorimage.shape
            x = linspace(0, len(andorimage[0])-1, len(andorimage[0]))
            y = linspace(0, len(andorimage)-1, len(andorimage))
            x,y = meshgrid(x, y)

            initial_guess = (andorimage.max()-andorimage.mean(),x.mean(),y.mean(),100,100,1000)
            print "Iteration: {}, Measurement: {}".format(iteration,measurement)
            try:
                popt, pcov = opt.curve_fit(func, (x,y), andorimage.ravel(), p0=initial_guess)
            except Exception as e:
                print "Fit failed: {}".format(e)
                popt = (NaN,)*len(initial_guess)
            data_fitted = func((x, y), *popt)
            print popt
            for i, el in enumerate(popt):
                popt_array[iteration-miniter,i]=el

            if (showGraphs):
                try:
                    self.showGraph(andorimage,popt,func)
                except Exception as e:
                    print "Graph failed. Did fit fail? Exception: {}".format(e)
        self.allpopts = popt_array
        self.poptmethod = 'fit_avgmeas'
        return popt_array


    # sumOverRegion: Instead of fitting, this function simply takes a region (in the form of xlimits,ylimits)
    #                and integrates the counts over that region. Useful in cases where you have small signal
    #                and can't easily fit the data.
    def sumOverRegion(self,avgMeasurements=True,errorbars=True):
        xlimit = self.xlimit
        ylimit = self.ylimit
        mytype=int64

        if (not avgMeasurements):
            sumArray = zeros((self.maxIter+1,self.maxMeas+1,1),dtype=mytype)
            for iteration in range(self.minIter,self.maxIter+1):
                for measurement in range(self.minMeas,self.maxMeas + 1,1):
                    andorimage = self.imagefunction(iteration,measurement)[xlimit[0]:xlimit[1],
                                                  ylimit[0]:ylimit[1]]
                    sumArray[iteration,measurement] = sum(andorimage)
        else:
            sumArray = zeros((self.maxIter+1,1+errorbars),dtype=mytype)
            for iteration in range(self.minIter, self.maxIter+1):
                andorimagemeas = zeros((self.maxMeas - self.minMeas + 1,xlimit[1] - xlimit[0],ylimit[1] - ylimit[0]),dtype=mytype)
                andorimagesum = zeros((self.maxMeas - self.minMeas + 1),dtype=mytype)
                for measurement in range(self.minMeas,self.maxMeas + 1,1):
                    andorimagemeas[measurement] = self.imagefunction(iteration,measurement)[xlimit[0]:xlimit[1],
                                                  ylimit[0]:ylimit[1]]
                    andorimagesum[measurement] = sum(andorimagemeas[measurement])
                if not errorbars:
                    andorimage = andorimagemeas.mean(0)
                    sumArray[iteration] = sum(andorimage)
                else:
                    sumArray[iteration] = array([andorimagesum.mean(),andorimagesum.std()])
        self.allpopts=sumArray
        self.poptmethod = 'sum'
        return sumArray


    def oneDIntegration(self,avgMeasurements=True,axis='y'):
        xlimit = self.xlimit
        ylimit = self.ylimit
        
        if axis=='y':
            intlen = ylimit[1] - ylimit[0]
            intaxis = 0
        elif axis=='x':
            intlen = xlimit[1] - xlimit[0]
            intaxis = 1
        else:
            print "Invalid value for axis in oneDIntegration. Should be 'x' or 'y'."
            return
        
        if (not avgMeasurements):
            sumArray = zeros((self.maxIter+1,self.maxMeas+1,intlen))
            for iteration in range(self.minIter,self.maxIter+1,1):
                for measurement in range(self.minMeas,self.maxMeas + 1,1):
                    andorimage = self.imagefunction(iteration,measurement)[xlimit[0]:xlimit[1],
                                                  ylimit[0]:ylimit[1]]
                    sumArray[iteration,measurement] = sum(andorimage,axis=intaxis)
        else:
            sumArray = zeros((self.maxIter+1,intlen))
            for iteration in range(self.minIter, self.maxIter+1):
                andorimagemeas = zeros((self.maxMeas - self.minMeas + 1,xlimit[1] - xlimit[0],ylimit[1] - ylimit[0]))
                for measurement in range(self.minMeas,self.maxMeas + 1,1):
                    andorimagemeas[measurement] = self.imagefunction(iteration,measurement)[xlimit[0]:xlimit[1],
                                                  ylimit[0]:ylimit[1]]
                andorimage = andorimagemeas.mean(0)
                sumArray[iteration] = sum(andorimage, axis=intaxis)
        return sumArray
                             

    def oneDIntegrationFitOneAxis(self,axis='y',showGraphs=False):
        func = self.oneD_Gaussian_W1
        sumArray = self.oneDIntegration(avgMeasurements=True,axis=axis)
        popt_array = zeros((self.maxIter-self.minIter+1,5))
        pcov_array = zeros((self.maxIter-self.minIter+1,5))
        for iteration in range(self.minIter,self.maxIter+1,1):
            x = linspace(0, len(sumArray[0])-1, len(sumArray[0]))
            ThisIteration = sumArray[iteration]
            minthisiter = min(ThisIteration[0],ThisIteration[-1])
            slopethisiter = (ThisIteration[-1]-ThisIteration[0])/(x[-1]-x[0])
            initial_guess = (ThisIteration.max()-ThisIteration.mean(),x.mean(),len(sumArray[0])/2,minthisiter,slopethisiter)
            print "Fitting 1D to Iteration {}".format(iteration)
            try:
                popt, pcov = opt.curve_fit(func, x, ThisIteration, p0=initial_guess, maxfev=100000)
            except Exception as e:
                print "Fit failed: {}".format(e)
                popt = (NaN,)*len(initial_guess)
            data_fitted = func(x, *popt)
            print popt
            for i, el in enumerate(popt):
                popt_array[iteration-self.minIter,i]=el
            pcov_array[iteration-self.minIter]=diag(pcov)

            if (showGraphs):
                try:
                    fig,ax = plt.subplots(1,1)
                    ax.hold(True)
                    ax.plot(x,ThisIteration,'ro',label="Integrated Counts")
                    ax.plot(x,data_fitted,label = "Fit to Gaussian".format(self.lifetime))
                    label = axis
                    ax.set_xlabel(label)
                    ax.set_ylabel('Integrated Counts (arb)')
                    plt.legend()
                    plt.ticklabel_format(axis='y',style='sci')
                    #ax.axis([0,max(xlin) * 1.1,0,max(amplitudes) * 1.1])
                    fig.show()
                except Exception as e:
                    print "Graph failed. Did fit fail? Exception: {}".format(e)
        self.allpopts = popt_array
        self.poptmethod = 'fit_avgmeas_1D'
        return popt_array, pcov_array
    
    def oneDIntegrationFit(self,showGraphs=False):
        popty, pcovy = self.oneDIntegrationFitOneAxis('y',showGraphs)
        poptx, pcovx = self.oneDIntegrationFitOneAxis('x',showGraphs)
        self.poptmethod = 'fit_avgmeas'

        #Parameters returned:
        # 0: Y amplitude
        # 1: X center
        # 2: Y center
        # 3: X width
        # 4: Y width
        # 5: X baseline
        # 6: Y baseline
        # 7: X amplitude
        # 8: X covariance
        # 9: Y covariance

        self.allpopts = array([[popty[i,0],poptx[i,1],popty[i,1],poptx[i,2],popty[i,2],poptx[i,3],popty[i,3],poptx[i,0]] for i in range(popty.shape[0])])
        self.perry = array([sqrt(pcovy[i]) for i in range(popty.shape[0])])
        self.perrx = array([sqrt(pcovx[i]) for i in range(popty.shape[0])])
        #self.allpopts = array([popty[0],poptx[1],popty[1],poptx[2],popty[2],poptx[3],popty[3],poptx[0]])
        return self.allpopts, sqrt(pcovx), sqrt(pcovy)
    
            
        

    # -------------- Pixel Size

    PixelSize = 4.7e-6   #From Jonathan's MATLAB program: in file Z:\Public\Hybrid\Useful Code\MATLAB\HybridAnalysisApp\User\configure.m

    # getBinMode: Given an open h5py File, this returns the bin Mode for the camera identified by cameraNum (by default the first camera)
    #            Returns 1 if 1x1 mode, 2 if 2x2 mode, 4 if 4x4 mode.
    def getBinMode(self):
        bn_tmp = self.resultsfile[(
            'settings/experiment/Andors/motors/motor{}/camera/binMode'.format(self.camnum)
        )]
        print(array(bn_tmp))
        self.binMode = (1,2,4)[array(bn_tmp)]
        return self.binMode

    def getTrapHoldTimes(self, min, max):
        return eval('' + self.resultsfile[('settings/experiment/independentVariables/Trap_Hold_time/function')])[
               min:max]

    # realPixelWidth: returns real-space width (in meters) of a pixel in the camera image, depending on the bin mode of the camera.
    def realPixelWidth(self,binMode):
        return binMode*self.PixelSize
    
    #binimg: bins an unbinned image (or you could bin a binned image again, whatever), 2x2
    def binimg(self,image):
        return array([[image[2*i,2*j]+image[2*i+1,2*j]+image[2*i,2*j+1]+image[2*i+1,2*j+1] 
                                 for j in range(image.shape[1]/2)] 
                                 for i in range(image.shape[0]/2)])

    # ---------------- Temperature fitting

    def fitCloud(self, cloudWidthsOrig, xlin, printtext="",fitorder=2):
        print printtext
        
        if fitorder==2:
            func = self.QuadraticFunctionNoB
            cloudWidths = cloudWidthsOrig
        elif fitorder==1:
            func = self.LineFunction
            cloudWidths = cloudWidthsOrig
            xlin = xlin**2

        initial_guess = [(cloudWidths[-1] - cloudWidths[0]) / (max(xlin) - min(xlin)),min(cloudWidths)**2]
        # print initial_guess

        try:
            poptlin,pcov = opt.curve_fit(func,xlin,cloudWidths,p0=initial_guess)
        except Exception as e:
            print "Fit failed: {}".format(e)
            poptlin = (NaN,) * len(initial_guess)
        # print poptlin
        # print xlin
        line = func(xlin,*poptlin)
        # print line

        mass = 2.20694657e-25
        kB = 1.38e-23

        T = poptlin[0] * mass / (4 * kB) * 1e6  # Temperature in microKelvin
        print "Temperature: {} microKelvin".format(T)

        fig,ax = plt.subplots(1,1)
        ax.hold(True)
        ax.plot(xlin,cloudWidths,'ro')
        ax.plot(xlin,line)
        if fitorder==2:
            plt.xlabel('Hold time (s)')
        elif fitorder==1:
            plt.xlabel('Hold time Squared (s$^2$)')
        plt.ylabel('Cloud Width Squared (m$^2$)')
        ax.axis([0,max(xlin) * 1.1,0,max(cloudWidths) * 1.1])
        fig.show()

        return T


    # getTemperaturesXY: performs fits for x and y MOT temperatures. Requires fitted parameter array and binMode as arguments
    def getTemperaturesXY(self,dropFrames=0,dropStartFrames=0,time=1e-3,fitorder=2):
        '''
        dropFrames; List of images to remove
        '''
        if dropFrames !=0:
            cloudWidthsX = (abs(self.allpopts[dropStartFrames:-dropFrames,3]) * self.realPixelWidth(self.binMode))**2
            cloudWidthsY = (abs(self.allpopts[dropStartFrames:-dropFrames,4]) * self.realPixelWidth(self.binMode))**2
        else:
            cloudWidthsX = (abs(self.allpopts[dropStartFrames:,3]) * self.realPixelWidth(self.binMode))**2
            cloudWidthsY = (abs(self.allpopts[dropStartFrames:,4]) * self.realPixelWidth(self.binMode))**2    
        print "Real Pixel Width: {} m".format(self.realPixelWidth(self.binMode)
                                            )
        xlin = eval(self.iVars[0][1])[dropStartFrames:dropStartFrames+len(cloudWidthsX)]*time   #self.getTrapHoldTimes(0,len(cloudWidthsX))/1000.0  # in seconds

        print "xlin: {}".format(xlin.shape)
        print "cloudWidthsX: {}".format(cloudWidthsX.shape)
        self.xT = self.fitCloud(cloudWidthsX,xlin,"X dimension fit",fitorder=fitorder)
        self.yT = self.fitCloud(cloudWidthsY,xlin,"Y dimension fit",fitorder=fitorder)
        return self.xT, self.yT
    
    # getTemperatures1D: performs fits for 1D MOT temperatures.
    def getTemperatures1D(self,dropFrames=0, axis='y'):
        '''
        dropFrames; List of images to remove
        '''
        
        if axis=='y' and self.poptmethod=='fit_avgmeas':
            cloudWidths = (abs(self.allpopts[:,4]) * self.realPixelWidth(self.binMode))**2
        elif axis=='x' and self.poptmethod=='fit_avgmeas':
            cloudWidths = (abs(self.allpopts[:,3]) * self.realPixelWidth(self.binMode))**2
        elif self.poptmethod=='fit_avgmeas_1D':
            cloudWidths = (abs(self.allpopts[:,2]) * self.realPixelWidth(self.binMode))**2
        print "Real Pixel Width: {} m".format(self.realPixelWidth(self.binMode)
                                            )
        xlin = self.getTrapHoldTimes(0,len(cloudWidths))/1000.0  # in seconds

        self.T = self.fitCloud(cloudWidths,xlin,"One dimension fit")
        return self.T


    # LoadFit: Calls loadImageStack to load Andor images, then calls fit_2DGauss_iterations_avg
    #          and returns parameters.
    def LoadFit(self, showgraphs=True, minIter=0, maxIter=-1, minMeas=0, maxMeas=-1):
        #if (len(self.allandorimages) == 0):
        #    self.loadImageStack()
        #print self.allandorimages.shape
        if (maxIter==-1):
            maxIter=self.maxIter
        if (maxMeas == -1):
            maxMeas = self.maxMeas
        self.allpopts = self.fit_2DGauss_iterations_avg(
            minIter,
            maxIter,
            minMeas,
            maxMeas,
            showgraphs)
        self.binMode = self.getBinMode()
        return self.allpopts

    def loadAbsorptionImages(self, maxmeas=-1, avgmeas=False):
        if (maxmeas != -1):
            self.maxMeas = maxmeas
        numshots = len(self.resultsfile['iterations/0/measurements/0/data/Andor_{}'.format(self.cameraID)])
        if (numshots != 2):
            print "Absorption images require 2 shots: one for the image and the other for background. {} shots in file".format(numshots)
        firstiter = self.imagefromh5(self.minIter,self.minMeas,0)
        if (avgmeas):
            nummeas=1
        else:
            nummeas=self.maxMeas - self.minMeas + 1
        allandorimages = zeros((self.maxIter - self.minIter + 1,nummeas,len(firstiter),len(firstiter[0])))
        for iteration in range(self.minIter,self.maxIter + 1,1):
            if (avgmeas):
                allandorimagesmeasback = zeros((self.maxMeas-self.minMeas+1,len(firstiter),len(firstiter[0])))
                allandorimagesmeassig = zeros((self.maxMeas-self.minMeas+1,len(firstiter),len(firstiter[0])))
            for measurement in range(self.minMeas,self.maxMeas + 1,1):
                if (avgmeas):
                    allandorimagesmeasback[measurement] = self.imagefromh5(iteration,measurement,1)
                    allandorimagesmeassig[measurement] = self.imagefromh5(iteration,measurement,0)
                else:
                    allandorimages[iteration,measurement] = self.imagefromh5(iteration,measurement,1) - self.imagefromh5(iteration,measurement,0)
            if (avgmeas):
                allandorimages[iteration,0] = allandorimagesmeasback.mean(0) - allandorimagesmeassig.mean(0)
        self.allandorimages = allandorimages
        return allandorimages

    #------------------------------   Image loading functions

    # self.imagefunction tells imageH5 how to load images from the HDF5 file. Currently two functions are set up (they follow this comment).
    #          One loads a single shot, and the other loads two shots per measurment and subtracts them to get an absorption image.
    #          If you want to add another method for loading images, define it here, then set self.imagefunction = self.yourfunction

    # imagefromh5: Loads an image from the HDF5 file for iteration,measurement,shot. shot is optional; assumed to be 0 if left out.
    def imagefromh5(self, iteration, measurement, shot=0):
        try:
            key_str = 'iterations/{}/measurements/{}/data/Andor_{}/shots/{}'.format(iteration,measurement,self.cameraID,shot)
            return self.resultsfile[key_str]
        except Exception as e:
            print("Exception : {} {}".format(e.__class__.__name__,e))
            print(key_str)
            raise
    
    def imageHamamatsu(self, iteration, measurement, shot=0):
        return self.resultsfile['iterations/{}/measurements/{}/data/Hamamatsu/shots/{}'.format(iteration,measurement,shot)][:]

    # gIFAbsorption: get Image Function Absorption. Uses imagefromh5 to get an absorption image for iteration,measurement.
    #                Assumes the background is shot 1 and signal is shot 0.
    def gIFAbsorption(self, iteration, measurement,shot=0):
        return array(self.imagefromh5(iteration,measurement,1)) - array(self.imagefromh5(iteration,measurement,0))

    # gIFBGSub: get Image Function Absorption. Uses imagefromh5 to get an absorption image for iteration,measurement.
    #                Assumes the background is shot 1 and signal is shot 0.
    def gIFBGsub(self, iteration, measurement,shot=0):
        return array(self.imagefromh5(iteration,measurement,0)) - array(self.imagefromh5(iteration,measurement,1))

    # gIFAbsorption: get Image Function Absorption. Uses imagefromh5 to get an absorption image for iteration,measurement.
    #                Assumes the background is shot 1 and signal is shot 0.
    def gHAbsorption(self, iteration, measurement,shot=0):
        return self.imageHamamatsu(iteration,measurement,1) - self.imageHamamatsu(iteration,measurement,0)
    
    # gIFAbsorption: get Image Function Absorption. Uses imagefromh5 to get an absorption image for iteration,measurement.
    #                Assumes the background is shot 1 and signal is shot 0.
    def gHBGSub(self, iteration, measurement,shot=0):
        return self.imageHamamatsu(iteration,measurement,0).astype(float) - self.imageHamamatsu(iteration,measurement,1).astype(float)

    # useAbsorptionImages: Use this function to tell the imageH5 object to use gIFAbsorption as its imagefunction.
    def useAbsorptionImages(self):
        self.imagefunction = self.gIFAbsorption

    # useSingleShot: Use this function to tell the imageH5 object to use imagefromh5 as its imagefunction.
    def useSingleShot(self):
        self.imagefunction = self.imagefromh5

    # LifetimeCurve: takes fitted parameters for andor images and uses them to calculate the
    #                trap lifetime.
    def LifetimeCurve(self,dropFrames=nan,poptmethod=-1):
        errorbars = False
        if isnan(dropFrames) or dropFrames==0:
            dropFrames = -(len(self.allpopts))
        if poptmethod==-1:
            poptmethod=self.poptmethod
        if (poptmethod == 'fit_avgmeas'):
            amplitudes = abs(self.allpopts[:-dropFrames,0])*abs(2*pi*self.allpopts[:-dropFrames,3]*self.allpopts[:-dropFrames,4])
        elif (poptmethod == 'fit'):
            amplitudes = abs(self.allpopts[:-dropFrames,:,0])*abs(2*pi*self.allpopts[:-dropFrames,:,3]*self.allpopts[:-dropFrames,:,4])
        elif (poptmethod == 'sum'):
            print self.allpopts.shape
            print dropFrames
            if self.allpopts.shape[1] == 2:
                print "Error bars on"
                errorbars = True
            else:
                errorbars = False
            amplitudes = self.allpopts[:-dropFrames,0]
        elif poptmethod == '1D_Xonly':
            amplitudes = abs(self.allpopts[:-dropFrames,7])*abs(2*pi*self.allpopts[:-dropFrames,3]*self.allpopts[:-dropFrames,3])
        xlin = eval(self.iVars[0][1])[:-dropFrames]/1000.0 #self.getTrapHoldTimes(0,len(amplitudes)) / 1000.0  # in seconds

        tau_IG = max(xlin)  #in case something weird happens, just make the initial guess large...
        for i, x in enumerate(xlin):   #initial guess for tau: find the first point that's less than 1/3 the initial guess for amplitude
            if amplitudes[i] - amplitudes[-1] < (amplitudes[0]-amplitudes[-1])/3.0:
                tau_IG = x
                break

        initial_guess = [amplitudes[0]-amplitudes[-1] , tau_IG, amplitudes[-1] ]
        try:
            poptlin,pcov = opt.curve_fit(self.DecayingExponential,xlin,amplitudes,p0=initial_guess)
        except Exception as e:
            print "Fit failed: {}".format(e)
            poptlin = (NaN,) * len(initial_guess)

        decayexp = self.DecayingExponential(xlin,*poptlin)

        self.lifetime = poptlin[1]
        print "Lifetime: {:.3} s".format(self.lifetime)

        fig,ax = plt.subplots(1,1)
        ax.hold(True)
        if not errorbars:
            ax.plot(xlin,amplitudes,'ro')
        else:
            ax.errorbar(xlin,amplitudes,yerr=self.allpopts[:-dropFrames,1],fmt='ro')
        ax.plot(xlin,decayexp,label = "Lifetime: {:.3} s".format(self.lifetime))
        plt.xlabel('Hold time (s)')
        plt.ylabel('Amplitude (arb)')
        plt.legend()
        #ax.axis([0,max(xlin) * 1.1,0,max(amplitudes) * 1.1])
        fig.show()

        return poptlin   #returns a list with elements (amplitude, tau, baseline)