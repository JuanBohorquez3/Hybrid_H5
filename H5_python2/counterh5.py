import os
import glob
import sys

from numpy import *
#from counter_analysis import counterData
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
from sklearn import mixture
import h5py
import pandas as pd
import seaborn as sbs
from mpl_toolkits.mplot3d import Axes3D
import imageh5 as i5
from imageh5 import seedpermute
import traceback
import copy##

gmix = mixture.GaussianMixture(n_components=2)

class counter():
    data=[]
    
    def __init__(self,folder='./',experiment=0,shots=None):
        imageH5 = i5.ImageH5('.', '1026', 0)
        try:
            shooting = True
            if shots is None:
                self.shots = int(imageH5.resultsfile['settings/experiment/independentVariables/Shots/function'].value)
            else:
                self.shots = shots
            print "Self.shots : {}".format(self.shots)
        except KeyError as e:
            shooting = False
            print 'Number of shots not found. Error : {}'.format(e)
        self.dir=os.getcwd().split(os.sep)[-1]
        self.filename = glob.glob(folder+'*.hdf5')[0]
        self.h5file = h5py.File(self.filename)
        self.measurements = self.h5file['settings/experiment/measurementsPerIteration'].value
        self.iterations = len(self.h5file['experiments/{}/iterations/'.format(experiment)].items())
        self.iVars=[]
        self.varPath=self.h5file['experiments/{}/iterations/0/variables'.format(experiment)]
        RObins = self.varPath['RO1_bins'].value
        ROdrops = self.varPath['RO1_drops'].value
        self.rawData=empty((self.iterations,self.measurements,self.shots,RObins))
        self.shotData=empty((self.iterations,self.measurements,self.shots))
        print self.rawData.shape
        if self.iterations > 1:
            for i in self.h5file['settings/experiment/independentVariables/'].iteritems(): 
                tmp=eval(i[1]['function'].value)
                if (type(tmp) == list) | (type(tmp) == ndarray) | (type(tmp) == tuple): self.iVars.append((i[0],i[1]['function'].value))
        for i in self.h5file['experiments/{}/iterations/'.format(experiment)].iteritems():
            for j in range(self.measurements):
                try:
                    temp=array(i[1]['measurements/{}/data/counter/data'.format(j)].value[0])
                except Exception as e:
                    print "Failed to load data."
                    print "Iteration {}, Measurement {}".format(i,j)
                    print "Exception: {}".format(e)
                    traceback.print_exc()

                if shooting:
                    # print "Temp length {}".format(len(temp))
                    for shot in range(self.shots):
                        # print "Shot {}".format(shot)
                        before = shot*(ROdrops+RObins)
                        # print "before {}".format(before)
                        try:
                            self.rawData[int(i[0]),j,shot]=temp[before+ROdrops:before+ROdrops+RObins]
                        except ValueError:
                            self.rawData[int(i[0]),j,shot] = array([0 for bob in self.rawData[int(i[0]),j,shot]])
                            for bob,t in enumerate(temp[before+ROdrops:before+ROdrops+RObins]):
                                self.rawData[int(i[0]),j,shot,bob] = t
                        self.shotData[int(i[0]), j,shot] = temp[before+ROdrops:before+ROdrops+RObins].sum()
                else :
                    self.rawData[int(i[0]),j,0]=temp[ROdrops:ROdrops+RObins]
                    self.rawData[int(i[0]),j,1]=temp[-RObins:]
                    self.shotData[int(i[0]),j,0]=temp[ROdrops:ROdrops+RObins].sum()
                    self.shotData[int(i[0]),j,1]=temp[-RObins:].sum()
                    self.shotData[int(i[0]),j,1]=temp[ROdrops*2+RObins:ROdrops*2+RObins*2].sum()
        if self.iterations > 1:
            df=0
            j=0
            for i in eval(self.iVars[0][1]):
                for shot in range(2):
                    d={self.iVars[0][0]: i, "Shot": shot, "Counts": self.shotData[j,:,shot]}
                    if df==0: 
                        df=1
                        self.DataFrame=pd.DataFrame(data=d)
                    else:       
                        tmpDF=pd.DataFrame(data=d)
                        self.DataFrame=self.DataFrame.append(tmpDF,ignore_index=True)
                j+=1
                        


                        
        self.cuts=[nan,nan,nan]
        self.cutserr = [nan,nan,nan]
        self.overlap=[nan,nan,nan]
        self.rload=[1,1,1]
        self.retention=[1,1]
        self.retentionerr = [1,1]
        
    def vplot(self,bw=None):
        plt.clf()
        sbs.violinplot(x=self.iVars[0][0],y='Counts', hue = 'Shot', data=self.DataFrame, split=True, inner='stick',bw=bw)


    def get_cuts(self,hbins=40,save_cuts=True,itr=0,set_initial_cut=None,force_retention_cut=False,force_cut2=False,plots=True):

        #=====================Fit Functions=================
        def intersection(A0,A1,m0,m1,s0,s1):
            return (m1*s0**2-m0*s1**2-sqrt(s0**2*s1**2*(m0**2-2*m0*m1+m1**2+2*log(A0/A1)*(s1**2-s0**2))))/(s0**2-s1**2)

        def intersectionerr(A0,A1,m0,m1,s0,s1,eA0,eA1,em0,em1,es0,es1) :
            #compute an error estimate for the cut location from the error 
            #estimates provided by the optimized fit, using the derivative of 
            #the intersection function
            #a useful constant
            radical = sqrt(s0**2*s1**2*((m0-m1)**2+2*log(A0/A1)*(s1**2-s0**2)))
            
            #print "difference of squares is : " + repr (s1**2-s0**2)
            #print "radical is : " + repr (radical)
            #partial derivatives
            dA0 = s0**2*s1**2/A0/radical
            dA1 = -s0**2*s1**2/A1/radical
            dm0 = ( -s1**2-s0**2*s1**2*(m0-m1) )/(s0**2-s1**2)/radical
            dm1 = ( s0**2+s1**2*s0**2*(m0-m1) )/(s0**2-s1**2)/radical
            ds0 = s0*s1**2*( 2*s1**2*(s1**2-s0**2)*log(A0/A1)+(m0-m1)*( (m0-m1)*(s0**2+s1**2)+2*radical ) )/(radical*(s0**2-s1**2)**2)
            ds1 = s1*s0**2*( 2*s0**2*(s0**2-s1**2)*log(A0/A1)-(m0-m1)*( (m0-m1)*(s0**2+s1**2)+2*radical ) )/(radical*(s0**2-s1**2)**2)
            errs = array([eA0,eA1,em0,em1,es0,es1])
            dervs = array([dA0,dA1,dm0,dm1,ds0,ds1])
            
            #print "the following is a list of partial derivatives : " + repr(dervs)
            cuterr = sqrt( sum((errs*dervs)**2) )
            #print "error estimate : " + repr(cuterr)
            return cuterr
            
        
        def area(A0,A1,m0,m1,s0,s1):
            return sqrt(pi/2)*(A0*s0+A0*s0*erf(m0/sqrt(2)/s0)+A1*s1+A1*s1*erf(m1/sqrt(2)/s1))

        # Normed Overlap for arbitrary cut point
        def overlap(xc,A0,A1,m0,m1,s0,s1):
            err0=A0*sqrt(pi/2)*s0*(1-erf((xc-m0)/sqrt(2)/s0))
            err1=A1*sqrt(pi/2)*s1*(erf((xc-m1)/sqrt(2)/s1)+erf(m1/sqrt(2)/s1))
            return (err0+err1)/area(A0,A1,m0,m1,s0,s1)

        # Relative Fraction in 1
        def frac(A0,A1,m0,m1,s0,s1):
            return 1/(1+A0*s0*(1+erf(m0/sqrt(2)/s0))/A1/s1/(1+erf(m1/sqrt(2)/s1)))
        
        def retentionErr(data,cut,cuterr):
            #compute the error in the retention by computing the upper and lower
            #bounds from the error estimate in the cut then taking the average 
            #of the difference from retention.
            retentionUB = ((data>(cut-cuterr))*1.0).sum()/len(data)
            retentionLB = ((data>(cut+cuterr))*1.0).sum()/len(data)
            retentionerr = abs((retentionUB-retentionLB)/2)
            return retentionerr

        def dblgauss(x,A0,A1,m0,m1,s0,s1):
            return A0*exp(-(x-m0)**2 / (2*s0**2)) +  A1*exp(-(x-m1)**2 / (2*s1**2))


        #====================================================


        #plt.close('all')
        titles=['Shot 1,Cut={:.2f}','Shot 2,Cut={:.2f}', 'PS Shot 2,Cut={:.2f}']
        if plots:
            f, axarr = plt.subplots(1,3,figsize=(12,6))
        for i in range(2):
            tmp=self.shotData[itr,:,i]
            gmix.fit(array([tmp]).transpose())
            est=[gmix.weights_.max()/10,gmix.weights_.min()/10,gmix.means_.min(),gmix.means_.max(),sqrt(gmix.means_.min()),sqrt(gmix.means_.max())]
            h = histogram(tmp,normed=True,bins=hbins)
            if plots:
                axarr[i].hist(tmp,bins=hbins,histtype='step',normed=True)
            try:
                if set_initial_cut is None:
                    popt, pcov = curve_fit(dblgauss, h[1][1:], h[0], est)
                    # popt=[A0,A1,m0,m1,s0,s1] : Absolute value
                    popt = abs(popt)
                    perr = sqrt(diag(pcov))
                    dpopt = copy.deepcopy(popt)
                    self.cuts[i] = intersection(*popt)
                    self.cutserr[i] = intersectionerr(*(append(dpopt, perr)))
                    self.overlap[i] = overlap(self.cuts[i], *popt)
                else:
                    self.cuts[i] = set_initial_cut
                    self.cutserr[i] = 0
                    self.overlap[i] = 0
                #Shot 0
                #======
                #Optimise Fit to Double-Gaussian Distribution Using Best-Fit Starting Values
                #popt,pcov = curve_fit(dblgauss,h[1][1:],h[0],est)
                #popt=[A0,A1,m0,m1,s0,s1] : Absolute value
                #popt=abs(popt)
                #perr=sqrt(diag(pcov))
                #dpopt = copy.deepcopy(popt)
                #print 'the following are fit parameters : \n'

                #print(popt)
                #print 'the following are fit errors : \n'
                #print(perr)
                #self.cuts[i]=intersection(*popt)
                #print "cut is : " + repr(self.cuts[i])
                #self.cutserr[i]=intersectionerr(*(append(dpopt,perr))
                if force_cut2 and i == 1:
                    self.cuts[i] = self.cuts[0]
                    self.cutserr[i] = self.cutserr[0]
                #self.overlap[i]=overlap(self.cuts[i],*popt)
                #plot the cut lines
                if plots:
                    axarr[i].plot(array([1,1])*self.cuts[i],axarr[i].get_ylim(),'k')
                if set_initial_cut is None:
                    if plots:
                        axarr[i].plot(h[1][1:] - .5, dblgauss(h[1][1:], *popt))
                        axarr[i].plot(array([1,1])*(self.cuts[i]+self.cutserr[i]),axarr[i].get_ylim(),'k')
                        axarr[i].plot(array([1,1])*(self.cuts[i]-self.cutserr[i]),axarr[i].get_ylim(),'k')
                        axarr[i].plot(h[1][1:]-.5,dblgauss(h[1][1:],*popt))
                    self.rload[i]=frac(*popt)
                else :
                    self.rload[i]=0

                #axarr[i].plot(h[1][1:]-.5,dblgauss(h[1][1:],*popt))
                if plots:
                    axarr[i].set_title(titles[i].format(self.cuts[i]))
            except RuntimeError as e :
                print "A RuntimeError occured while plotting shot " + repr(i)
                print sys.exc_info()
                self.cuts[i]=nan
                self.cutserr[i]=nan
                self.rload[i]=nan
                if force_cut2 and i == 1:
                    self.cuts[i] = self.cuts[0]
                    self.cutserr[i] = self.cutserr[0]
        #self.retention[0]=self.rload[1]/self.rload[0]
        #not yet implemented as it is not currently being used
        self.retentionerr[0] = nan


        self.cut()
        tmp=self.shotData[itr,where((self.shotData[itr,:,0]>self.cuts[0])*1.0==1.0),1][0]
        print len(tmp)
        if set_initial_cut is None:
            gmix.fit(array([tmp]).transpose())
            est=[gmix.weights_.max()/10,gmix.weights_.min()/10,gmix.means_.min(),gmix.means_.max(),sqrt(gmix.means_.min()),sqrt(gmix.means_.max())]
        h = histogram(tmp,normed=True,bins=hbins)
        if plots:
            axarr[2].hist(tmp,bins=hbins,histtype='step',normed=True)
        try:
                #Shot 0
                #======
                #Optimise Fit to Double-Gaussian Distribution Using Best-Fit Starting Values
            if set_initial_cut is None:
                popt,pcov = curve_fit(dblgauss,h[1][1:],h[0],est)
                #popt=[A0,A1,m0,m1,s0,s1] : Absolute value
                popt=abs(popt)
            if force_retention_cut:
                self.cuts[2] = self.cuts[1]
                self.cutserr[2]= self.cutserr[1]
            else:
                if set_initial_cut is None:
                    perr=sqrt(diag(pcov))
                    dpopt = copy.deepcopy(popt)
                #print 'the following are fit parameters : \n'
                #print(popt)
                #print 'the following are fit errors : \n'
                #print(perr)
                    self.cuts[2]=intersection(*popt)
                #print "cut is : " + repr(self.cuts[2])
                    self.cutserr[2]=intersectionerr(*(append(dpopt,perr)))
                #print "cut error is : " + repr(self.cutserr[2])
                else:
                    self.cuts[2]=set_initial_cut
                    self.cutserr[2]=0

            #plot the cut lines
            if plots:
                axarr[2].plot(array([1,1])*self.cuts[2],axarr[2].get_ylim(),'k')
            if set_initial_cut is None:
                self.overlap[2]=overlap(self.cuts[2],*popt)
                self.rload[2] = frac(*popt)
                if plots:
                    axarr[2].plot(h[1][1:]-.5,dblgauss(h[1][1:],*popt))
                    axarr[2].plot(array([1,1])*(self.cuts[2]+self.cutserr[2]),axarr[2].get_ylim(),'k')
                    axarr[2].plot(array([1,1])*(self.cuts[2]-self.cutserr[2]),axarr[2].get_ylim(),'k')
                    axarr[2].plot(h[1][1:]-.5,dblgauss(h[1][1:],*popt))
            else:
                self.overlap[2]=0
                self.rload[2]=0
        except RuntimeError:
            print "A RuntimeError occured while plotting shot " + repr(2)
            print sys.exc_info()
            self.cuts[2]=nan
            self.cutserr[2] = nan
            self.rload[2]=nan
        if force_retention_cut:
            self.cuts[2] = self.cuts[1]
            self.cutserr[2] = self.cutserr[1]


        if plots:
            axarr[2].set_title(titles[2].format(self.cuts[2]))

        self.retention[1]=((tmp>self.cuts[2])*1.0).sum()/len(tmp)
        self.retentionerr[1] = retentionErr(tmp,self.cuts[2],self.cutserr[2])
        
        print "cuts, cutserr, rload, retention, retentionerr : "
        print self.cuts,self.cutserr,self.rload, self.retention, self.retentionerr
        if plots:
            plt.suptitle(self.dir[:19].replace('_',' ')+' Calibration , Load Frac={:.1%}, Retention={:.1%}'.format(self.rload[0],self.retention[1]),size=16)
            plt.show()
            plt.savefig('../'+self.dir+'_CalCutPlots.png')
        if save_cuts==True: savetxt('../'+self.dir[:19]+'_Cuts.txt' ,concatenate((self.cuts,self.rload,self.retention)))
        return (self.cuts, self.rload,self.retention,self.retentionerr,self.overlap)

    def cut(self):
        rshape=self.rawData.shape
        out=zeros((rshape[0],rshape[1],2))
        #print out.shape

        if isnan(self.cuts[0]): self.load_cuts()
        for i in range(2):
            out[:,:,i]=self.shotData[:,:,i]>self.cuts[i]
        savetxt(self.dir+'_Binarized.txt',out.reshape(rshape[0],rshape[1]*2),header='Rload_cut={}'.format(self.retention[1]),fmt='%i')
        self.binData=out
        return out

    def load_cuts(self):
        files=glob.glob('../*_Cuts.txt')
        files.sort()
        try:
            tmp = loadtxt(files[-1])
            print tmp
            self.cuts=tmp[0:3]
            self.rload=tmp[3:6]
            self.retention=tmp[6:8]
            print self.cuts,self.rload, self.retention
        except IndexError:
            print 'Bad Cut File!'

    def hist3D(self,shot=0):
        mx=int(self.shotData[:,:,shot].max())
        plt.close('all')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(self.iterations):
            tmp=histogram(self.shotData[i,:,shot],bins=mx/2)
            ax.bar(tmp[1][:-1],tmp[0],zs=eval(self.iVars[0][1])[i],zdir='y',alpha=.5)

        ax.set_xlabel('cts')
        ax.set_zlabel('N')
        #ax.set_ylabel(self.varVar[0])
        ax.view_init(elev=30,azim=38)
        #ax.set_title(self.varName[0]+' (' +self.varVar[0]+ ') Bin='+str(self.dt)+'ms')
        plt.show()
        #plt.savefig('Hist3D_'+self.dir+'_Bin_'+str(self.dt)+'ms.png')
        
    def binparscan(self, cut0=30, cut1=30, fit=True):
        

        
        if len(self.iVars)>1:
        
            p=(self.shotData[:,:,1] > cut1).sum(1)*1./(self.shotData[:,:,0] > cut0).sum(1)
            err=sqrt((1-p)*p/(self.shotData[:,:,0] > cut0).sum(1))
            it_scan = 0
            if self.iVars[0][0]=='shelve_state': it_scan = 1
            it_arr = eval(self.iVars[it_scan][1])
            it_len = len(it_arr)
            
            plt.clf()
            
            plt.errorbar(it_arr,p[:it_len],yerr=err[:it_len],label='F=3')
            plt.errorbar(it_arr,p[it_len:],yerr=err[it_len:],label='F=4')
            plt.xlabel(self.iVars[it_scan][0])     
            plt.ylabel('Retention')
            
            
        if len(self.iVars) == 1:
            p=(self.shotData[:,:,1] > cut1).sum(1)*1./(self.shotData[:,:,0] > cut0).sum(1)
            err=sqrt((1-p)*p/(self.shotData[:,:,0] > cut0).sum(1))
            it_arr = eval(self.iVars[0][1])
            it_len = len(it_arr)
            
            
            
            plt.clf()
            plt.errorbar(it_arr,p,yerr=err, label = 'Data',fmt='.')
        
            if fit:
                sin_func = lambda t,f,A,b: abs(A)*sin(pi*f*(t))**2 + b
                popt,pcov = curve_fit(sin_func, it_arr, p, sigma=err, p0=[10,p.max()-p.min(),p.min()])
                plt.plot(linspace(it_arr[0],it_arr[-1],1000), sin_func(linspace(it_arr[0],it_arr[-1],1000),*popt),label = 'fit: frequency = {:.5f} kHz'.format(popt[0]))
                print 'A={:.5},b={:.5},piTime={}'.format(popt[1],popt[2],1/(2*popt[0]))
            plt.xlabel(self.iVars[0][0])

        plt.ylabel('Retention')
        plt.legend()
        return p,err
        
        


    
    
    def fitparscan(self,hbins=30,tr=0):
    # Currently just plots the overlap between Atom and Background
    #        
    #        
            #=====================Fit Functions=================
        def intersection(A0,A1,m0,m1,s0,s1):
            return (m1*s0**2-m0*s1**2-sqrt(s0**2*s1**2*(m0**2-2*m0*m1+m1**2+2*log(A0/A1)*(s1**2-s0**2))))/(s0**2-s1**2)
    
        def area(A0,A1,m0,m1,s0,s1):
            return sqrt(pi/2)*(A0*s0+A0*s0*erf(m0/sqrt(2)/s0)+A1*s1+A1*s1*erf(m1/sqrt(2)/s1))
    
        # Normed Overlap for arbitrary cut point
        def overlap(xc,A0,A1,m0,m1,s0,s1):    
            err0=A0*sqrt(pi/2)*s0*(1-erf((xc-m0)/sqrt(2)/s0))
            err1=A1*sqrt(pi/2)*s1*(erf((xc-m1)/sqrt(2)/s1)+erf(m1/sqrt(2)/s1))
            return (err0+err1)/area(A0,A1,m0,m1,s0,s1) 
    
        # Relative Fraction in 1
        def frac(A0,A1,m0,m1,s0,s1):    
            return 1/(1+A0*s0*(1+erf(m0/sqrt(2)/s0))/A1/s1/(1+erf(m1/sqrt(2)/s1)))
        
        def dblgauss(x,A0,A1,m0,m1,s0,s1):
            return A0*exp(-(x-m0)**2 / (2*s0**2)) +  A1*exp(-(x-m1)**2 / (2*s1**2))
        
    
        #====================================================
    
    
        plt.close('all')
        out=zeros((2,self.shotData.shape[0]))
        perr=[]
        fracout=zeros((2,self.shotData.shape[0]))
        for i in range(out.shape[1]):
            for shot in range(2):
                tmp=self.shotData[i,:,shot]
                gmix.fit(array([tmp]).transpose())
                est=[gmix.weights_.max()/10,gmix.weights_.min()/10,gmix.means_.min(),gmix.means_.max(),sqrt(gmix.means_.min()),sqrt(gmix.means_.max())]
                h = histogram(tmp,normed=True,bins=hbins)
                try:
                    #Shot 0
                    #======           
                    #Optimise Fit to Double-Gaussian Distribution Using Best-Fit Starting Values
                    popt,pcov = curve_fit(dblgauss,h[1][1:],h[0],est)
                    #popt=[A0,A1,m0,m1,s0,s1] : Absolute value
                    popt=abs(popt) 
                    
                    xc=intersection(*popt)
                    
                    perr.append(overlap(xc,*sqrt(diag(pcov))))
                    
                    if isnan(xc): 
                        print 'Bad Cut on Shot: {} Iteration: {}'.format(shot,i)
                        out[shot,i]=nan
                        fracout[shot,i]=nan
                        #perr.append(1)
                    else:
                        out[shot,i]=overlap(xc,*popt)
                        if frac(*popt) < 1:
                            fracout[shot,i]=frac(*popt)
                        else:
                            fracout[shot,i]=1/frac(*popt)
                except (RuntimeError,RuntimeWarning):
                    print 'Bad fit on Shot: {} Iteration: {}'.format(shot,i)
                    out[shot,i]=nan
                    fracout[shot,i]=nan
        fracout=fracout[1]/fracout[0]
        print "Fracout=",fracout
        print "Perr[0]=", perr[::2]
        print "out=", out
        fracout[where(fracout>=1)]=nan
        if len(self.iVars)>1: 
            out=out.reshape(len(self.varSpace[1]),len(self.varSpace[0]))
            if tr == 0:out=out.transpose()
            f,axarr = plt.subplots(2,sharex=True,figsize=(12,6))
            labels=['{} = {:.3f}'.format(self.varName[(tr+1)%2],self.varSpace[(tr+1)%2][i]) for i in range(len(self.varSpace[(tr+1)%2]))]
            axarr[0].errorbar(self.varSpace[tr],out,'.')
            axarr[0].set_title('Raw Data')
            axarr[1].errorbar(self.varSpace[tr],out/self.retention[1],'.')
            axarr[1].set_title('Scaled Data')
            axarr[1].set_xlabel(self.varName[tr])
        else:
            varSpace=eval(self.iVars[0][1])
            f,axarr = plt.subplots(3,sharex=True,figsize=(12,6))
            labels=[self.iVars[0][0]]
            axarr[0].plot(varSpace,out[0],'.')#, yerr = perr[::2])
            axarr[0].set_title('Background-Atom Overlap %: Shot 0')
            axarr[1].plot(varSpace,out[1],'.')#, yerr = perr[1::2])
            axarr[1].set_title('Background-Atom Overlap %: Shot 1')
            axarr[2].plot(varSpace,fracout,'.')#, yerr = perr[1::2])
            axarr[2].set_title('Post Experiment Retention')
            axarr[2].set_xlabel(self.iVars[0][0])
            
                                    
        plt.suptitle('Double Gaussian Fitted Parameter Scan')    
        #plt.legend(labels,fontsize='small')
        plt.show()
        plt.savefig(self.dir+'_FitParScan1D.png')
        
        
        return out
        
    def select_single(self, hbins=40, save_cuts=True, itr=0, diagnostic_shot=1, data_shot=0, force_cut = False, cut = None):
        """
        To be used during measurements where we would like to select for measurements in which a
        single atom was found in the diagnostic_shot.

        For the given iteration, determines the background-single-atom-cutoff, determines which measurements had a
        single atom in the second shot, and returns all shot data for the first and second shots from those measurements

        This function does not plot data for the sake of time and memory

        :param hbins: Integer, how many bins to divide the data into
        :param diagnostic_shot: Integer, shot to be used to select data
        :param save_cuts: Boolean, should the cuts for this iteration be saved in the cuts file?
        :param itr: Integer, iteration to be looked at
        :param force_cut: Boolean, determines whether to force a particular cut based on input parameters or determine a
            cut using the fit functions
        :param cut: Integer, cut to be fixed in place of fit parameters
        :return: Returns a dictionary with a bunch of information about the cut
        """

        # =====================Fit Functions=================
        def intersection(A0, A1, m0, m1, s0, s1):
            return (m1 * s0 ** 2 - m0 * s1 ** 2 - sqrt(
                s0 ** 2 * s1 ** 2 * (m0 ** 2 - 2 * m0 * m1 + m1 ** 2 + 2 * log(A0 / A1) * (s1 ** 2 - s0 ** 2)))) / (
                               s0 ** 2 - s1 ** 2)

        def intersectionerr(A0, A1, m0, m1, s0, s1, eA0, eA1, em0, em1, es0, es1):
            # compute an error estimate for the cut location from the error
            # estimates provided by the optimized fit, using the derivative of
            # the intersection function

            # a useful constant
            radical = sqrt(s0 ** 2 * s1 ** 2 * ((m0 - m1) ** 2 + 2 * log(A0 / A1) * (s1 ** 2 - s0 ** 2)))

            # print "difference of squares is : " + repr (s1**2-s0**2)
            # print "radical is : " + repr (radical)
            # partial derivatives
            dA0 = s0 ** 2 * s1 ** 2 / A0 / radical
            dA1 = -s0 ** 2 * s1 ** 2 / A1 / radical
            dm0 = (-s1 ** 2 - s0 ** 2 * s1 ** 2 * (m0 - m1)) / (s0 ** 2 - s1 ** 2) / radical
            dm1 = (s0 ** 2 + s1 ** 2 * s0 ** 2 * (m0 - m1)) / (s0 ** 2 - s1 ** 2) / radical
            ds0 = s0 * s1 ** 2 * (2 * s1 ** 2 * (s1 ** 2 - s0 ** 2) * log(A0 / A1) + (m0 - m1) * (
                        (m0 - m1) * (s0 ** 2 + s1 ** 2) + 2 * radical)) / (radical * (s0 ** 2 - s1 ** 2) ** 2)
            ds1 = s1 * s0 ** 2 * (2 * s0 ** 2 * (s0 ** 2 - s1 ** 2) * log(A0 / A1) - (m0 - m1) * (
                        (m0 - m1) * (s0 ** 2 + s1 ** 2) + 2 * radical)) / (radical * (s0 ** 2 - s1 ** 2) ** 2)
            errs = array([eA0, eA1, em0, em1, es0, es1])
            dervs = array([dA0, dA1, dm0, dm1, ds0, ds1])

            # print "the following is a list of partial derivatives : " + repr(dervs)
            cuterr = sqrt(sum((errs * dervs) ** 2))
            # print "error estimate : " + repr(cuterr)
            return cuterr

        def area(A0, A1, m0, m1, s0, s1):
            return sqrt(pi / 2) * (
                        A0 * s0 + A0 * s0 * erf(m0 / sqrt(2) / s0) + A1 * s1 + A1 * s1 * erf(m1 / sqrt(2) / s1))

        # Normed Overlap for arbitrary cut point
        def overlap(xc, A0, A1, m0, m1, s0, s1):
            err0 = A0 * sqrt(pi / 2) * s0 * (1 - erf((xc - m0) / sqrt(2) / s0))
            err1 = A1 * sqrt(pi / 2) * s1 * (erf((xc - m1) / sqrt(2) / s1) + erf(m1 / sqrt(2) / s1))
            return (err0 + err1) / area(A0, A1, m0, m1, s0, s1)

        # Relative Fraction in 1
        def frac(A0, A1, m0, m1, s0, s1):
            return 1 / (1 + A0 * s0 * (1 + erf(m0 / sqrt(2) / s0)) / A1 / s1 / (1 + erf(m1 / sqrt(2) / s1)))

        def retentionErr(data, cut, cuterr):
            # compute the error in the retention by computing the upper and lower
            # bounds from the error estimate in the cut then taking the average
            # of the difference from retention.
            retentionUB = ((data > (cut - cuterr)) * 1.0).sum() / len(data)
            retentionLB = ((data > (cut + cuterr)) * 1.0).sum() / len(data)
            retentionerr = abs((retentionUB - retentionLB) / 2)
            return retentionerr

        def dblgauss(x, A0, A1, m0, m1, s0, s1):
            return A0 * exp(-(x - m0) ** 2 / (2 * s0 ** 2)) + A1 * exp(-(x - m1) ** 2 / (2 * s1 ** 2))

        tmp = self.shotData[itr, :, diagnostic_shot]
        gmix.fit(array([tmp]).transpose())
        est = [gmix.weights_.max() / 10, gmix.weights_.min() / 10, gmix.means_.min(), gmix.means_.max(),
               sqrt(gmix.means_.min()), sqrt(gmix.means_.max())]
        h = histogram(tmp, normed=True, bins=hbins)
        try:
            # Shot 0
            # ======
            # Optimise Fit to Double-Gaussian Distribution Using Best-Fit Starting Values
            if force_cut and cut is not None:
                self.cuts[diagnostic_shot] = cut
                self.cutserr[diagnostic_shot] = 0
                self.overlap[diagnostic_shot] = 0
            else:
                popt, pcov = curve_fit(dblgauss, h[1][1:], h[0], est)
                # popt=[A0,A1,m0,m1,s0,s1] : Absolute value
                popt = abs(popt)
                perr = sqrt(diag(pcov))
                dpopt = copy.deepcopy(popt)
                self.cut[diagnostic_shot] = intersection(*popt)
                self.cutserr[diagnostic_shot] = intersectionerr(*(append(dpopt, perr)))
                self.overlap[diagnostic_shot] = overlap(self.cuts[diagnostic_shot], *popt)
            self.rload[diagnostic_shot] = ((tmp > self.cuts[diagnostic_shot])*1.0).sum()/len(tmp)
            rl_plus = ((tmp > self.cuts[diagnostic_shot]+self.cutserr[diagnostic_shot])*1.0).sum()/len(tmp)
            rl_minus = ((tmp > self.cuts[diagnostic_shot]-self.cutserr[diagnostic_shot])*1.0).sum()/len(tmp)
            rl_err = abs(rl_plus-rl_minus)/2

        except RuntimeError:
            print "A RuntimeError occured while plotting shot " + repr(diagnostic_shot) + " iteration " + repr(itr)
            print sys.exc_info()
            self.cuts[diagnostic_shot] = nan
            self.cutserr[diagnostic_shot] = nan
            self.rload[diagnostic_shot] = nan
            rl_err = nan

        # data array = [raw data with an atom, raw data without an atom]
        there_be_atoms = where(self.shotData[itr, :, diagnostic_shot] > self.cuts[diagnostic_shot])
        no_atoms = where(self.shotData[itr, :, diagnostic_shot] <= self.cuts[diagnostic_shot])

        raw_data = {"Single Atom": self.rawData[itr, there_be_atoms, data_shot, :][0],
                    "No Atom": self.rawData[itr, no_atoms, data_shot, :][0]}

        shot_data = {"Single Atom": self.shotData[itr, there_be_atoms, data_shot][0],
                     "No Atom": self.shotData[itr, no_atoms, data_shot][0]}

        raw_diag = {"Single Atom": self.rawData[itr, there_be_atoms, diagnostic_shot, :][0],
                    "No Atom": self.rawData[itr, no_atoms, diagnostic_shot, :][0]}

        shot_diag = {"Single Atom": self.shotData[itr, there_be_atoms, diagnostic_shot][0],
                     "No Atom": self.shotData[itr, no_atoms, diagnostic_shot][0]}

        atom_indexes = {"Single Atom": there_be_atoms,
                        "No Atom": no_atoms}

        status = {"Cut": self.cuts[diagnostic_shot],
                  "Cut Error": self.cutserr[diagnostic_shot],
                  "Raw Data": raw_data,
                  "Shot Data": shot_data,
                  "Raw Diagnostic": raw_diag,
                  "Shot Diagnostic": shot_diag,
                  "Atom Locations": atom_indexes,
                  "Full Histogram": h,
                  "Load Fraction": self.rload[diagnostic_shot],
                  "Load Error Bars": rl_err}
        return status