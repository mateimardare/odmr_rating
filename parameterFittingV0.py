import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import time
import threading

import matplotlib.pyplot as plt

'''define the fit functions here'''
#Gauss
def Gaussfkt_1D(x, A, x0, sigx, B):
    return A * np.exp(-(x - x0) ** 2 / (2 * sigx ** 2)) + B

def Gaussfkt_2D(x, y, A, x0, y0, sigx, sigy, B):
    return A * np.exp(-(x - x0) ** 2 / (2 * sigx ** 2) - (y - y0) ** 2 / (2 * sigy ** 2)) + B

def _Gaussfkt_2D_flat(M, *args):
    x, y = M
    return Gaussfkt_2D(x, y, *args)


#Sine
def SINE_func(t, A, T, t_0, y_0):
    return (A*0.5)*np.sin((2*np.pi/T)*(np.subtract(t,t_0))) + y_0

def SINEphase_func(phi, A, phi_0, y_0):
    return (A*0.5)*np.sin((np.pi/180)*(np.subtract(phi,phi_0))) + y_0

def SINEphase_func_k(phi, A, k, phi_0, y_0):
    return (A*0.5)*np.sin(k*(np.pi/180)*(np.subtract(phi,phi_0))) + y_0


#exponential decreasing Sine
def RABI_func(t, A, T, t_0, tau, y_0):
    return (A*0.5)*np.cos((2*np.pi/T)*(np.subtract(t,t_0)))*np.exp(
        -1*np.divide(t,tau))+y_0
# get pi pulses
def get_rabi_T(popt, perr):
    '''From Rabi fit optimal parameter dictionary gets rabi period. Return dict with period + error.'''
    T = {}
    T_err = {}
    for file_name in popt.keys():
        T[file_name] = popt[file_name][1]
        T_err[file_name] = perr[file_name][1]
    return T, T_err

def get_pi_pulse(T, T_err, t_0, t_0_err):
    '''From rabi period dict extract pi pulse length. Returns dict with pi pulse for each file name (dict keys)'''
    pi = {}
    pi_err = {}
    for file_name in T.keys():
        pi[file_name] = np.divide(T[file_name], 2) - t_0[file_name]
        pi_err[file_name] = np.sqrt(np.add(np.power(np.divide(T_err[file_name], 2),2), np.power(t_0_err[file_name],2)))
    return pi, pi_err

def get_pi_half_pulse(T, T_err, t_0, t_0_err):
    '''From rabi period dict extract pi/2 pulse length. Returns dict with pi pulse for each file name (dict keys)'''
    pi_half = {}
    pi_half_err = {}
    for file_name in T.keys():
        pi_half[file_name] = np.divide(T[file_name], 4) - t_0[file_name]
        pi_err[file_name] = np.sqrt(np.add(np.power(np.divide(T_err[file_name], 4),2), np.power(t_0_err[file_name],2)))
    return pi_half, pi_half_err

def get_three_pi_half_pulse(T, T_err, t_0, t_0_err):
    '''From rabi period dict extract 3*pi/2 pulse length. Returns dict with pi pulse for each file name (dict keys)'''
    three_pi_half = {}
    three_pi_half_err = {}
    for file_name in T.keys():
        three_pi_half[file_name] = np.multiply(3, np.divide(T[file_name], 2)) - t_0[file_name]
        three_pi_half_err[file_name] = np.sqrt(np.add(np.power(np.multiply(3,np.divide(T_err[file_name], 2)),2), np.power(t_0_err[file_name],2)))
    return three_pi_half, three_pi_half_err

def get_rabi_freq(T, T_err):
    f = {}
    f_err = {}
    for file_name in T.keys():
        f[file_name] = np.divide(1, T[file_name])
        f_err[file_name] = np.divide(1*T_err[file_name], np.power(T[file_name],2))
    return f, f_err

#Ramsey
def ramsey_func(x, A, sig_x,  B):
    return A * np.exp(- x**2 / (2 * sig_x**2)) + B

def ramsey_mod_func(x, A, sig_x, k ,f0, B):
    return A * np.exp(- x**2 / (2 * sig_x**2)) * ( 1 - k * np.sin(2*np.pi * f0 * x/1000) ** 2 ) + B

#hahn_echo
def hahn_echo_func(x, A, T2, n, k ,f0):
    # return A * np.exp(- (2*x/T2)**n ) * (1-k*np.sin(np.pi*f0/1000*x)**2 * np.sin(np.pi*f1/1000*x)**2 )
    return A * np.exp(- ( x/1000 / T2) ** n) * ( 1 - k * np.sin(2*np.pi * f0 * x/1000) ** 2 )

#linar
def lin_fit(x, a, b):
    return a*x+b

#exponential
def exp_func(x, a, b):
    return a*np.exp(b*x)


def intensity(power, I_s, P_s, p0, y0):
    return np.divide(np.multiply(I_s, power+p0), np.add(power+p0, P_s+p0)) + y0

def quad_func(x, a, b, c):
    return np.multiply(a, np.power(x,2)) + np.multiply(b,x) + c

#Lorentz
def peak(f, A, gamma, f_peak):
    return np.divide(A*np.power((gamma/2),2),np.power(np.subtract(f,f_peak),2)+np.power((gamma/2),2))


def odmr_dips_fit(f, *fit_params_guess):
    # the number of fit params determines the number of dips to be fitted
    # fit params is an array of the form
    # [A_1, A_2, ..., gamma_1, gamma_2, ..., f_peak_1, f_peak_2, ...]
    # where A is a guess on the peak amplitude,
    # Gamma is a guess on the peak FWHM
    # f_peak is a guess on the frequency location of the peak

    peak_num = int((len(fit_params_guess)-1)/3)

    A = fit_params_guess[0:peak_num]

    f_peak = fit_params_guess[peak_num:2*peak_num]

    gamma = fit_params_guess[2*peak_num:3*peak_num]

    y_0 = fit_params_guess[3*peak_num]

    fit_curve = np.zeros(len(f))+y_0
    for i in range(0, peak_num):
        fit_curve = fit_curve - peak(f, A[i], gamma[i], f_peak[i])

    return fit_curve


class paramFITclass:
    def __init__(self, LoadDict=None):
        if LoadDict == None:
            self.type = 'none'

        elif isinstance(LoadDict, dict):
            self.update_from_dict(LoadDict)

        else:
            raise TypeError("invalid input")

    def __del__(self):
        pass

        #############################################################################################################

    # dict functions
    def to_dict(self):
        return self.__dict__
        # use the following syntax to be able to replace certain values
        # this_dict = self.__dict__.copy()
        # if 'XXX' in this_dict:   #remove XXX from dict
        #     del this_dict['XXX']
        # return this_dict

    def update_from_dict(self, newdict):
        self.__dict__.update(newdict)

    def fitQuadratic(self, X,Y, varY=None, usebounds = None, startparam = None, printresults=True):
        # X, Y (must have the same length)
        # varY is the variance of Y (used to weight the fit if given, must have the same length as X,Y)
        # a*x^2 + b*x + c
        # startparam['a']
        # startparam['b']
        # startparam['c']

        # usebounds['a'] : (min,max)
        # usebounds['b'] : (min,max)
        # usebounds['c'] : (min,max)

        self.type = 'fitQuadratic'
        self.X = X
        self.Y = Y
        self.varY = varY

        # let the automatic guess always do the job
        self.startparam = {}
        self.startparam['a'] = 1
        self.startparam['b'] = 0
        self.startparam['c'] = 0

        self.usebounds = {}
        self.usebounds['a'] = (-np.inf, np.inf)
        self.usebounds['b'] = (-np.inf, np.inf)
        self.usebounds['c'] = (-np.inf, np.inf)

        if (startparam is not None):
            for thiskey in startparam.keys():
                try:
                    self.startparam[thiskey] = startparam[thiskey]
                except:
                    print('WARNING: paramFITclass: the key in the startparam is not valid : ' + thiskey)
                    print('         allowed ones:')
                    for tmp in self.startparam.keys():
                        print('                ' + tmp)

        if (usebounds is not None):
            for thiskey in usebounds.keys():
                try:
                    self.usebounds[thiskey] = usebounds[thiskey]
                except:
                    print('WARNING: paramFITclass: the key in the usebounds is not valid : ' + thiskey)
                    print('         allowed ones:')
                    for tmp in self.usebounds.keys():
                        print('                ' + tmp)

        p0 = [self.startparam['a'], self.startparam['b'], self.startparam['c']]

        bounds = ((self.usebounds['a'][0], self.usebounds['b'][0], self.usebounds['c'][0]),
                  (self.usebounds['a'][1], self.usebounds['b'][1], self.usebounds['c'][1]))

        if varY is None:
            self.popt, self.pcov = curve_fit(quad_func, X, Y, p0=p0, bounds=bounds, maxfev=10000)
        else:
            self.popt, self.pcov = curve_fit(quad_func, X, Y, p0=p0, bounds=bounds, sigma = varY, maxfev=10000)

        pstd = []
        for k in range(len(self.popt)):
            pstd.append(np.sqrt(self.pcov[k,k]))

        self.fitres = {}
        self.fitres['a']  = self.popt[0]
        self.fitres['a_u'] = pstd[0]
        self.fitres['b'] = self.popt[1]
        self.fitres['b_u'] = pstd[1]
        self.fitres['c'] = self.popt[2]
        self.fitres['c_u'] = pstd[2]

        self.fitres['x_opt'] = -self.fitres['b'] / 2 / self.fitres['a']
        self.fitres['x_opt_u'] = self.fitres['x_opt'] * np.sqrt((self.fitres['b_u'] / self.fitres['b']) ** 2 + (self.fitres['a_u'] / self.fitres['a']) ** 2)

        self.fitres['y_opt'] = quad_func(self.fitres['x_opt'], *self.popt)
        self.fitres['y_opt_u'] = np.sqrt( self.fitres['x_opt']**4 * self.fitres['a_u']**2
                                          + self.fitres['x_opt']**2 * self.fitres['b_u']**2
                                          + self.fitres['c_u']**2)

        if printresults:
            print("fit results:")
            for thiskey in self.fitres.keys():
                print("              %(key)s      : %(1)s " % {"key":thiskey ,"1": str(self.fitres[thiskey])})

        self.Xplot = np.linspace(X[0],X[-1],10*len(X))
        self.Yplot = quad_func(self.Xplot, *self.popt)

        return self.fitres, self.Xplot, self.Yplot, quad_func(self.Xplot, *p0)

    def fit_exponential(self, X,Y, varY=None, usebounds = None, startparam = None, printresults=True):
        # X, Y (must have the same length)
        # varY is the variance of Y (used to weight the fit if given, must have the same length as X,Y)
        # exp_func(x, a, b)
        # startparam['a'] :  Amplitude
        # startparam['b'] : peak position

        # usebounds['a'] : (min,max) of Amplitude (the same bounds are used for all peaks)
        # usebounds['b'] : (min,max) of peak position (the same bounds are used for all peaks)

        self.type = 'fit_exponential'
        self.X = X
        self.Y = Y
        self.varY = varY

        # let the automatic guess always do the job
        self.startparam = {}
        self.startparam['a'] = np.amax(Y)
        self.startparam['b'] = 0

        self.usebounds = {}
        self.usebounds['a'] = (-np.inf, np.inf)
        self.usebounds['b'] = (-np.inf, np.inf)

        if (startparam is not None):
            for thiskey in startparam.keys():
                try:
                    self.startparam[thiskey] = startparam[thiskey]
                except:
                    print('WARNING: paramFITclass: the key in the startparam is not valid : ' + thiskey)
                    print('         allowed ones:')
                    for tmp in self.startparam.keys():
                        print('                ' + tmp)

        if (usebounds is not None):
            for thiskey in usebounds.keys():
                try:
                    self.usebounds[thiskey] = usebounds[thiskey]
                except:
                    print('WARNING: paramFITclass: the key in the usebounds is not valid : ' + thiskey)
                    print('         allowed ones:')
                    for tmp in self.usebounds.keys():
                        print('                ' + tmp)


        # build it again together
        p0 = [self.startparam['a'], self.startparam['b']]

        bounds = ((self.usebounds['a'][0], self.usebounds['b'][0]),
                  (self.usebounds['a'][1], self.usebounds['b'][1]))

        if varY is None:
            self.popt, self.pcov = curve_fit(exp_func, X, Y, p0=p0, bounds=bounds, maxfev=10000)
        else:
            self.popt, self.pcov = curve_fit(exp_func, X, Y, p0=p0, bounds=bounds, sigma = varY, maxfev=10000)

        pstd = []
        for k in range(len(self.popt)):
            pstd.append(np.sqrt(self.pcov[k,k]))

        self.fitres = {}
        self.fitres['a']  = self.popt[0]
        self.fitres['a_u'] = pstd[0]
        self.fitres['b'] = self.popt[1]
        self.fitres['b_u'] = pstd[1]

        if printresults:
            print("fit results:")
            for thiskey in self.fitres.keys():
                print("              %(key)s      : %(1)s " % {"key":thiskey ,"1": str(self.fitres[thiskey])})

        self.Xplot = np.linspace(X[0],X[-1],10*len(X))
        self.Yplot = exp_func(self.Xplot, *self.popt)

        return self.fitres, self.Xplot, self.Yplot, exp_func(self.Xplot, *p0)


    def fitLorentzianSingleDip(self, X,Y, varY=None, usebounds = None, startparam = None, printresults=True):
        # X, Y (must have the same length)
        # varY is the variance of Y (used to weight the fit if given, must have the same length as X,Y)
        # startparam['A'] :  Amplitude
        # startparam['x0'] : peak position
        # startparam['gamma'] : linewidth
        # startparam['B'] : offset

        # usebounds['A'] : (min,max) of Amplitude (the same bounds are used for all peaks)
        # usebounds['x0'] : (min,max) of peak position (the same bounds are used for all peaks)
        # usebounds['gamma'] : (min,max) of linewidth (the same bounds are used for all peaks)
        # usebounds['B'] : (min,max) of offset

        self.type = 'fitLorentzianSingleDip'
        self.X = X
        self.Y = Y
        self.varY = varY

        # let the automatic guess always do the job

        self.startparam = {}
        self.startparam['B'] = np.amax(Y)
        self.startparam['A'] = np.amax(Y) - np.amin(Y)
        tmp = np.sum(Y < (self.startparam['B'] - self.startparam['A'] / 2))
        if tmp < 1:
            tmp = 1
        self.startparam['x0'] = X[np.argmin(Y)]
        self.startparam['gamma'] = 0.5 * np.absolute((X[-1] - X[0]) / len(X)) * tmp


        self.usebounds = {}
        self.usebounds['B'] = (0,4*self.startparam['B'])
        self.usebounds['A'] = (0,4*self.startparam['A'])
        self.usebounds['x0'] = (X[0], X[-1])
        self.usebounds['gamma'] = ( (X[1]-X[0])/2, X[-1]-X[0])



        # if the number of peaks is correct, then use the rest of defined startparam values
        if (startparam is not None):
            for thiskey in startparam.keys():
                try:
                    self.startparam[thiskey] = startparam[thiskey]
                except:
                    print('WARNING: paramFITclass: the key in the startparam is not valid : ' + thiskey)
                    print('         allowed ones:')
                    for tmp in self.startparam.keys():
                        print('                ' + tmp)

        if (usebounds is not None):
            for thiskey in usebounds.keys():
                try:
                    self.usebounds[thiskey] = usebounds[thiskey]
                except:
                    print('WARNING: paramFITclass: the key in the usebounds is not valid : ' + thiskey)
                    print('         allowed ones:')
                    for tmp in self.usebounds.keys():
                        print('                ' + tmp)


        # build it again together
        p0 = [self.startparam['A'], self.startparam['x0'], self.startparam['gamma'], self.startparam['B'] ]

        bounds = ((self.usebounds['A'][0], self.usebounds['x0'][0], self.usebounds['gamma'][0], self.usebounds['B'][0]),
                  (self.usebounds['A'][1], self.usebounds['x0'][1], self.usebounds['gamma'][1], self.usebounds['B'][1]))

        if varY is None:
            self.popt, self.pcov = curve_fit(odmr_dips_fit, X, Y, p0=p0, bounds=bounds, maxfev=10000)
        else:
            self.popt, self.pcov = curve_fit(odmr_dips_fit, X, Y, p0=p0, bounds=bounds, sigma = varY, maxfev=10000)

        pstd = []
        for k in range(len(self.popt)):
            pstd.append(np.sqrt(self.pcov[k,k]))

        self.fitres = {}
        self.fitres['A']  = self.popt[0]
        self.fitres['A_u'] = pstd[0]
        self.fitres['x0'] = self.popt[1]
        self.fitres['x0_u'] = pstd[1]
        self.fitres['gamma'] = self.popt[2]
        self.fitres['gamma_u'] = pstd[2]
        self.fitres['B'] = self.popt[3]
        self.fitres['B_u'] = pstd[3]

        if printresults:
            print("fit results:")
            for thiskey in self.fitres.keys():
                print("              %(key)s      : %(1)s " % {"key":thiskey ,"1": str(self.fitres[thiskey])})

        self.Xplot = np.linspace(X[0],X[-1],10*len(X))
        self.Yplot = odmr_dips_fit(self.Xplot, *self.popt)

        return self.fitres, self.Xplot, self.Yplot, odmr_dips_fit(self.Xplot, *p0)


    def fitLorentzian(self, X,Y, varY=None, usebounds = None, startparam = None, printresults=True):
        # X, Y (must have the same length)
        # varY is the variance of Y (used to weight the fit if given, must have the same length as X,Y)
        # startparam['A'] :  Amplitudes (list)
        # startparam['x0'] : peak positions (list)
        # startparam['gamma'] : linewidth (list)
        # startparam['B'] : offset

        # usebounds['A'] : (min,max) of Amplitude (the same bounds are used for all peaks)
        # usebounds['x0'] : (min,max) of peak position (the same bounds are used for all peaks)
        # usebounds['gamma'] : (min,max) of linewidth (the same bounds are used for all peaks)
        # usebounds['B'] : (min,max) of offset

        self.type = 'fitLorentzian'
        self.X = X
        self.Y = Y
        self.varY = varY

        # let the automatic guess always do the job
        p0,bounds,peak_num = self.init_params_fit_ODMR(X, Y, height=None, prominence=None)
        self.startparam = {}
        self.startparam['peak_num'] = peak_num
        self.startparam['A']  = p0[0:peak_num]
        self.startparam['x0'] = p0[peak_num:2 * peak_num]
        self.startparam['gamma'] = p0[2 * peak_num:3 * peak_num]
        self.startparam['B'] = p0[-1]

        self.usebounds = {}
        self.usebounds['A'] = (bounds[0][0],bounds[1][0])
        self.usebounds['x0'] = (bounds[0][peak_num], bounds[1][peak_num])
        self.usebounds['gamma'] = (bounds[0][2 * peak_num], bounds[1][2 * peak_num])
        self.usebounds['B'] = (bounds[0][-1], bounds[1][-1])

        # if startparam is defined, then check if the required number of peaks are found
        Fit_valid = True
        # if startparam is not None:
        #     if not (self.startparam['peak_num'] == startparam['peak_num']):
        #         Fit_valid = False

        # if the number of peaks is correct, then use the rest of defined startparam values
        if (startparam is not None) and Fit_valid:
            for thiskey in startparam.keys():
                try:
                    self.startparam[thiskey] = startparam[thiskey]
                except:
                    print('WARNING: paramFITclass: the key in the startparam is not valid : ' + thiskey)
                    print('         allowed ones:')
                    for tmp in self.startparam.keys():
                        print('                ' + tmp)

        if (startparam is not None) and Fit_valid:
            for thiskey in startparam.keys():
                try:
                    self.startparam[thiskey] = startparam[thiskey]
                except:
                    print('WARNING: paramFITclass: the key in the startparam is not valid : ' + thiskey)
                    print('         allowed ones:')
                    for tmp in self.startparam.keys():
                        print('                ' + tmp)

        if (usebounds is not None) and Fit_valid:
            for thiskey in usebounds.keys():
                try:
                    self.usebounds[thiskey] = [thiskey]
                except:
                    print('WARNING: paramFITclass: the key in the usebounds is not valid : ' + thiskey)
                    print('         allowed ones:')
                    for tmp in self.usebounds.keys():
                        print('                ' + tmp)


        # build it again together
        p0 = np.append(np.concatenate((self.startparam['A'], self.startparam['x0'], self.startparam['gamma'])), self.startparam['B'])

        tmp0 = list(np.full(peak_num, self.usebounds['A'][0]))
        tmp0.extend(list(np.full(peak_num, self.usebounds['x0'][0])))
        tmp0.extend(list(np.full(peak_num, self.usebounds['gamma'][0])))
        tmp0.append(self.usebounds['B'][0])
        tmp1 = list(np.full(peak_num, self.usebounds['A'][1]))
        tmp1.extend(list(np.full(peak_num, self.usebounds['x0'][1])))
        tmp1.extend(list(np.full(peak_num, self.usebounds['gamma'][1])))
        tmp1.append(self.usebounds['B'][1])
        bounds = (tuple(tmp0) , tuple(tmp1))

        if varY is None:
            self.popt, self.pcov = curve_fit(odmr_dips_fit, X, Y, p0=p0, bounds=bounds, maxfev=10000)
        else:
            self.popt, self.pcov = curve_fit(odmr_dips_fit, X, Y, p0=p0, bounds=bounds, sigma = varY, maxfev=10000)

        pstd = []
        for k in range(len(self.popt)):
            pstd.append(np.sqrt(self.pcov[k,k]))

        self.fitres = {}
        self.fitres['fit_valid'] = Fit_valid
        self.fitres['A']  = self.popt[0:self.startparam['peak_num']]
        self.fitres['A_u'] = pstd[0:self.startparam['peak_num']]
        self.fitres['x0'] = self.popt[self.startparam['peak_num']:2 * self.startparam['peak_num']]
        self.fitres['x0_u'] = pstd[self.startparam['peak_num']:2 * self.startparam['peak_num']]
        self.fitres['gamma'] = self.popt[2 * self.startparam['peak_num']:3 * self.startparam['peak_num']]
        self.fitres['gamma_u'] = pstd[2 * self.startparam['peak_num']:3 * self.startparam['peak_num']]
        self.fitres['B'] = self.popt[-1]

        if printresults:
            print("fit results:")
            for thiskey in self.fitres.keys():
                print("              %(key)s      : %(1)s " % {"key":thiskey ,"1": str(self.fitres[thiskey])})

        self.Xplot = np.linspace(X[0],X[-1],10*len(X))
        self.Yplot = odmr_dips_fit(self.Xplot, *self.popt)

        return self.fitres, self.Xplot, self.Yplot, odmr_dips_fit(self.Xplot, *p0)

    def init_params_fit_ODMR(self, x, y, height=None, prominence=None, width = 3, distance=3):
        y0 = np.mean(y)

        if height == None:
            height = 2*np.sqrt(np.max(y))
        try:
            if prominence == None:
                peaks, _ = find_peaks(y0 - y, height=height,
                                      distance=distance, width=width)
            else:
                peaks, _ = find_peaks(
                    y0 - y, height=height, distance=distance, prominence=prominence, width=width)
        except:
            pass
        res = x[1] - x[0]

        A = np.full(len(peaks), np.abs(y0 - np.min(y)))
        Amin = np.full(len(peaks), 0)
        Amax = np.full(len(peaks), 2*y0)

        dip_pos = []
        fwhm = []
        for i in range(0, len(peaks)):
            dip_pos.append(x[int(peaks[i])])
            fwhm.append(2*res)

        dip_pos_min = np.full(len(peaks), x[0])
        dip_pos_max = np.full(len(peaks), x[-1])
        fwhm_min = np.full(len(peaks), res)
        fwhm_max = np.full(len(peaks), (x[-1]-x[0])/2)

        p0 = np.append(np.concatenate((A, dip_pos, fwhm)), y0)
        bounds = (tuple(np.append(np.concatenate((Amin, dip_pos_min, fwhm_min)), 0)),tuple(np.append(np.concatenate((Amax, dip_pos_max, fwhm_max)), 4*y0)))

        return p0,bounds, len(peaks)

    def monotonic(self,x):
        dx = np.diff(x)
        return np.all(dx <= 0) or np.all(dx >= 0)



    def fit1DgaussianPeak(self, X,Y ,varY=None, usebounds = None, startparam = None, printresults=True):
        # X, Y (must have the same length)
        # varY is the variance of Y (used to weight the fit if given, must have the same length as X,Y)
        # startparam['A'] :  Amplitude
        # startparam['x0'] : peak position
        # startparam['sigx'] : standard deviation
        # startparam['B'] : offset

        # usebounds['A'] : (min,max) of Amplitude
        # usebounds['x0'] : (min,max) of peak position
        # usebounds['sigx'] : (min,max) of standard deviation
        # usebounds['B'] : (min,max) of offset

        self.type = 'fit1DgaussianPeak'
        self.X = X
        self.Y = Y
        self.varY = varY

        # define start parameters, if startparam is defined in the function then overwrite the defined ones later
        B_start = np.amin(Y)
        A_start = np.amax(Y) - B_start
        tmp = np.sum(Y > (B_start + A_start / 2))
        if tmp < 1:
            tmp = 1
        sigx_start = 0.5 * np.absolute((X[-1] - X[0]) / len(X)) * tmp
        self.startparam = {'A': A_start,
                           'x0': X[np.argmax(Y)],
                           'sigx': sigx_start,
                           'B': B_start}

        if startparam is not None:
            for thiskey in startparam.keys():
                try:
                    self.startparam[thiskey] = startparam[thiskey]
                except:
                    print('WARNING: paramFITclass: the key in the startparam is not valid : ' + thiskey)
                    print('         allowed ones:')
                    for tmp in self.startparam.keys():
                        print('                ' + tmp)

        p0 = [self.startparam['A'], self.startparam['x0'], self.startparam['sigx'], self.startparam['B']]

        # define the bound parameters, if usebounds is defined in the function then overwrite the defined ones later
        self.usebounds = {'A': [0, 4 * self.startparam['A']],
                          'x0': [X[0], X[-1]],
                          'sigx': [0, X[-1]],
                          'B': [0, 2*np.amax(Y)]}

        if usebounds is not None:
            for thiskey in usebounds.keys():
                try:
                    self.usebounds[thiskey] = usebounds[thiskey]
                except:
                    print('WARNING: paramFITclass: the key in the usebounds is not valid : ' + thiskey)
                    print('         allowed ones:')
                    for tmp in self.usebounds.keys():
                        print('                ' + tmp)

        bounds = ((self.usebounds['A'][0], self.usebounds['x0'][0], self.usebounds['sigx'][0], self.usebounds['B'][0]),
                  (self.usebounds['A'][1], self.usebounds['x0'][1], self.usebounds['sigx'][1], self.usebounds['B'][1]))

        if printresults:
            print(p0)
            print(bounds)

        if varY is None:
            self.popt, self.pcov = curve_fit(Gaussfkt_1D, X, Y, p0=p0, bounds=bounds)
        else:
            self.popt, self.pcov = curve_fit(Gaussfkt_1D, X, Y, p0=p0, bounds=bounds, sigma = varY)

        self.fitres = {'A': self.popt[0],'A_u':np.sqrt(self.pcov[0, 0]),
                       'x0': self.popt[1], 'x0_u': np.sqrt(self.pcov[1, 1]),
                       'sigx': self.popt[2], 'sigx_u': np.sqrt(self.pcov[2, 2]),
                       'B': self.popt[3], 'B_u': np.sqrt(self.pcov[3, 3])}

        if printresults:
            print("fit results:")
            for thiskey in self.fitres.keys():
                print("              %(key)s      : %(1)f " % {"key":thiskey ,"1": self.fitres[thiskey]})

        self.Xplot = np.linspace(X[0],X[-1],10*len(X))
        self.Yplot = Gaussfkt_1D(self.Xplot, self.popt[0], self.popt[1], self.popt[2], self.popt[3])

        return self.fitres, self.Xplot, self.Yplot



    def fitRABI(self, X,Y ,varY=None, usebounds = None, startparam = None, printresults=True):
        # X, Y (must have the same length)
        # varY is the variance of Y (used to weight the fit if given, must have the same length as X,Y)
        # def RABI_func(t, A, T, t_0, tau, y_0):
        #     return (A*0.5)*np.cos((2*np.pi/T)*(np.subtract(t,t_0)))*np.exp(
        #         -1*np.divide(t,tau)**(1))+y_0

        # startparam['A'] :   Amplitude
        # startparam['T'] :   period (in same units as X)
        # startparam['t_0'] : time offset (in same units as X)
        # startparam['tau'] : decoherence decay time (in same units as X)
        # startparam['y_0'] : offset

        # usebounds['A'] :   (min,max) of  Amplitude
        # usebounds['T'] :   (min,max) of period (in same units as X)
        # usebounds['t_0'] : (min,max) of time offset (in same units as X)
        # usebounds['tau'] : (min,max) of decoherence decay time (in same units as X)
        # usebounds['y_0'] : (min,max) of offset

        self.type = 'fitRABI'
        self.X = X
        self.Y = Y
        self.varY = varY

        # define start parameters, if startparam is defined in the function then overwrite the defined ones later
        self.startparam = { 'A':np.abs(np.amax(Y) - np.amin(Y)) / 2,
                            'T':X[-1]/2,
                            't_0':0,
                            'tau': X[-1]*4,
                            'y_0':np.mean(Y)}

        if startparam is not None:
            for thiskey in startparam.keys():
                try:
                    self.startparam[thiskey] = startparam[thiskey]
                except:
                    print('WARNING: paramFITclass: the key in the startparam is not valid : ' + thiskey)
                    print('         allowed ones:')
                    for tmp in self.startparam.keys():
                        print('                '+tmp)

        p0 = [self.startparam['A'], self.startparam['T'], self.startparam['t_0'], self.startparam['tau'], self.startparam['y_0']]

        # define the bound parameters, if usebounds is defined in the function then overwrite the defined ones later
        self.usebounds = {'A': [0, 4*self.startparam['A']],
                          'T': [0, X[-1]*10],
                          't_0': [0,X[-1]*4],
                          'tau': [0, X[-1]*1000],
                          'y_0': [0,self.startparam['y_0']]}

        if usebounds is not None:
            for thiskey in usebounds.keys():
                try:
                    self.usebounds[thiskey] = usebounds[thiskey]
                except:
                    print('WARNING: paramFITclass: the key in the usebounds is not valid : ' + thiskey)
                    print('         allowed ones:')
                    for tmp in self.usebounds.keys():
                        print('                '+tmp)

        bounds = ((self.usebounds['A'][0], self.usebounds['T'][0], self.usebounds['t_0'][0], self.usebounds['tau'][0], self.usebounds['y_0'][0]),
                  (self.usebounds['A'][1], self.usebounds['T'][1], self.usebounds['t_0'][1], self.usebounds['tau'][1], self.usebounds['y_0'][1]))

        if printresults:
            print(p0)
            print(bounds)

        if varY is None:
            self.popt, self.pcov = curve_fit(RABI_func, X, Y, p0=p0, bounds=bounds)
        else:
            self.popt, self.pcov = curve_fit(RABI_func, X, Y, p0=p0, bounds=bounds, sigma = varY)

        self.fitres = {'A': self.popt[0],'A_u': np.sqrt(self.pcov[0, 0]),
                       'T': self.popt[1], 'T_u': np.sqrt(self.pcov[1, 1]),
                       't_0': self.popt[2], 't_0_u': np.sqrt(self.pcov[2, 2]),
                       'tau': self.popt[3], 'tau_u': np.sqrt(self.pcov[3, 3]),
                       'y_0': self.popt[4], 'y_0_u': np.sqrt(self.pcov[4, 4])}

        if printresults:
            print("fit results:")
            for thiskey in self.fitres.keys():
                print("              %(key)s      : %(1)f " % {"key":thiskey ,"1": self.fitres[thiskey]})

        self.Xplot = np.linspace(X[0],X[-1],10*len(X))
        self.Yplot = RABI_func(self.Xplot, self.popt[0], self.popt[1], self.popt[2], self.popt[3], self.popt[4])

        return self.fitres, self.Xplot, self.Yplot

    def fitSINEtime(self, X,Y ,varY=None, usebounds = None, startparam = None, printresults=True):
        # X, Y (must have the same length)
        # varY is the variance of Y (used to weight the fit if given, must have the same length as X,Y)
        # def SINE_func(t, A, T, t_0, y_0):
        #     return (A * 0.5) * np.sin((2 * np.pi / T) * (np.subtract(t, t_0))) + y_0

        # startparam['A'] :   Amplitude
        # startparam['T'] :   period (in same units as X)
        # startparam['t_0'] : time offset (in same units as X)
        # startparam['y_0'] : offset

        # usebounds['A'] :   (min,max) of  Amplitude
        # usebounds['T'] :   (min,max) of period (in same units as X)
        # usebounds['t_0'] : (min,max) of time offset (in same units as X)
        # usebounds['y_0'] : (min,max) of offset

        self.type = 'fitSINEtime'
        self.X = X
        self.Y = Y
        self.varY = varY


        # define start parameters, if startparam is defined in the function then overwrite the defined ones later
        self.startparam = { 'A':np.abs(np.amax(Y) - np.amin(Y)) / 2,
                            'T':X[-1]/2,
                            't_0':0,
                            'y_0':np.mean(Y)}

        if startparam is not None:
            for thiskey in startparam.keys():
                try:
                    self.startparam[thiskey] = startparam[thiskey]
                except:
                    print('WARNING: paramFITclass: the key in the startparam is not valid : ' + thiskey)
                    print('         allowed ones:')
                    for tmp in self.startparam.keys():
                        print('                '+tmp)

        p0 = [self.startparam['A'], self.startparam['T'], self.startparam['t_0'], self.startparam['y_0']]

        # define the bound parameters, if usebounds is defined in the function then overwrite the defined ones later
        self.usebounds = {'A': [0, 4*self.startparam['A']],
                          'T': [0, X[-1]*10],
                          't_0': [0,X[-1]*4],
                          'y_0': [0,self.startparam['y_0']]}

        if usebounds is not None:
            for thiskey in usebounds.keys():
                try:
                    self.usebounds[thiskey] = usebounds[thiskey]
                except:
                    print('WARNING: paramFITclass: the key in the usebounds is not valid : ' + thiskey)
                    print('         allowed ones:')
                    for tmp in self.usebounds.keys():
                        print('                '+tmp)

        bounds = ((self.usebounds['A'][0], self.usebounds['T'][0], self.usebounds['t_0'][0], self.usebounds['y_0'][0]),
                  (self.usebounds['A'][1], self.usebounds['T'][1], self.usebounds['t_0'][1], self.usebounds['y_0'][1]))

        if printresults:
            print(p0)
            print(bounds)


        if varY is None:
            self.popt, self.pcov = curve_fit(SINE_func, X, Y, p0=p0, bounds=bounds)
        else:
            self.popt, self.pcov = curve_fit(SINE_func, X, Y, p0=p0, bounds=bounds, sigma = varY)

        self.fitres = {'A': self.popt[0],'A_u':np.sqrt(self.pcov[0, 0]),
                       'T': self.popt[1], 'T_u': np.sqrt(self.pcov[1, 1]),
                       't_0': self.popt[2], 't_0_u': np.sqrt(self.pcov[2, 2]),
                       'y_0': self.popt[3], 'y_0_u': np.sqrt(self.pcov[3, 3])}

        if printresults:
            print("fit results:")
            for thiskey in self.fitres.keys():
                print("              %(key)s      : %(1)f " % {"key":thiskey ,"1": self.fitres[thiskey]})

        self.Xplot = np.linspace(X[0],X[-1],10*len(X))
        self.Yplot = SINE_func(self.Xplot, self.popt[0], self.popt[1], self.popt[2], self.popt[3])

        return self.fitres, self.Xplot, self.Yplot


    def fitSINEphase_k(self, X,Y ,varY=None, usebounds = None, startparam = None, printresults=True):
        # X, Y (must have the same length)
        # varY is the variance of Y (used to weight the fit if given, must have the same length as X,Y)
        # def SINEphase_func(phi, A, phi_0, y_0):
        #     return (A * 0.5) * np.sin((np.pi / 180) * (np.subtract(phi, phi_0))) + y_0

        # startparam['A'] :   Amplitude
        # startparam['k'] :   frequency like multiplicator sin(k * (phi-ph0))
        # startparam['phi_0'] : phase offset (in same units as X)
        # startparam['y_0'] : offset

        # usebounds['A'] :   (min,max) of  Amplitude
        # usebounds['k'] : (min,max) of multiplicator sin(k * (phi-ph0))
        # usebounds['phi_0'] : (min,max) of phase offset (in same units as X)
        # usebounds['y_0'] : (min,max) of offset

        self.type = 'fitSINEphase_k'
        self.X = X
        self.Y = Y
        self.varY = varY

        # define start parameters, if startparam is defined in the function then overwrite the defined ones later
        self.startparam = { 'A':np.abs(np.amax(Y) - np.amin(Y)) / 2,
                            'k': 1,
                            'phi_0':0,
                            'y_0':np.mean(Y)}

        if startparam is not None:
            for thiskey in startparam.keys():
                try:
                    self.startparam[thiskey] = startparam[thiskey]
                except:
                    print('WARNING: paramFITclass: the key in the startparam is not valid : ' + thiskey)
                    print('         allowed ones:')
                    for tmp in self.startparam.keys():
                        print('                '+tmp)

        p0 = [self.startparam['A'], self.startparam['k'], self.startparam['phi_0'], self.startparam['y_0']]

        # define the bound parameters, if usebounds is defined in the function then overwrite the defined ones later
        self.usebounds = {'A': [0, 4*self.startparam['A']],
                          'k': [0, np.inf],
                          'phi_0': [-720,720],
                          'y_0': [-2*np.amax(Y),2*np.amax(Y)]}

        if usebounds is not None:
            for thiskey in usebounds.keys():
                try:
                    self.usebounds[thiskey] = usebounds[thiskey]
                except:
                    print('WARNING: paramFITclass: the key in the usebounds is not valid : ' + thiskey)
                    print('         allowed ones:')
                    for tmp in self.usebounds.keys():
                        print('                '+tmp)

        bounds = ((self.usebounds['A'][0],self.usebounds['k'][0], self.usebounds['phi_0'][0], self.usebounds['y_0'][0]),
                  (self.usebounds['A'][1],self.usebounds['k'][1], self.usebounds['phi_0'][1], self.usebounds['y_0'][1]))

        if printresults:
            print(p0)
            print(bounds)


        if varY is None:
            self.popt, self.pcov = curve_fit(SINEphase_func_k, X, Y, p0=p0) #, bounds=bounds
        else:
            self.popt, self.pcov = curve_fit(SINEphase_func_k, X, Y, p0=p0, bounds=bounds, sigma = varY)

        self.fitres = {'A': self.popt[0],'A_u':np.sqrt(self.pcov[0, 0]),
                       'k': self.popt[1], 'k_u': np.sqrt(self.pcov[1, 1]),
                       'phi_0': self.popt[2], 'phi_0_u': np.sqrt(self.pcov[2, 2]),
                       'y_0': self.popt[3], 'y_0_u': np.sqrt(self.pcov[3, 3])}

        if printresults:
            print("fit results:")
            for thiskey in self.fitres.keys():
                print("              %(key)s      : %(1)f " % {"key":thiskey ,"1": self.fitres[thiskey]})

        # create the plot
        self.Xplot = np.linspace(X[0],X[-1],10*len(X))
        self.Yplot = SINEphase_func_k(self.Xplot, self.popt[0], self.popt[1], self.popt[2], self.popt[3])

        return self.fitres, self.Xplot, self.Yplot, SINEphase_func_k(self.Xplot, *p0)

    def fitSINEphase(self, X, Y, varY=None, usebounds=None, startparam=None, printresults=True):
        # X, Y (must have the same length)
        # varY is the variance of Y (used to weight the fit if given, must have the same length as X,Y)
        # def SINEphase_func(phi, A, phi_0, y_0):
        #     return (A * 0.5) * np.sin((np.pi / 180) * (np.subtract(phi, phi_0))) + y_0

        # startparam['A'] :   Amplitude
        # startparam['phi_0'] : phase offset (in same units as X)
        # startparam['y_0'] : offset

        # usebounds['A'] :   (min,max) of  Amplitude
        # usebounds['phi_0'] : (min,max) of phase offset (in same units as X)
        # usebounds['y_0'] : (min,max) of offset

        self.type = 'fitSINEphase'
        self.X = X
        self.Y = Y
        self.varY = varY

        # define start parameters, if startparam is defined in the function then overwrite the defined ones later
        self.startparam = {'A': np.abs(np.amax(Y) - np.amin(Y)) / 2,
                           'phi_0': 0,
                           'y_0': np.mean(Y)}

        if startparam is not None:
            for thiskey in startparam.keys():
                try:
                    self.startparam[thiskey] = startparam[thiskey]
                except:
                    print('WARNING: paramFITclass: the key in the startparam is not valid : ' + thiskey)
                    print('         allowed ones:')
                    for tmp in self.startparam.keys():
                        print('                ' + tmp)

        p0 = [self.startparam['A'], self.startparam['phi_0'], self.startparam['y_0']]

        # define the bound parameters, if usebounds is defined in the function then overwrite the defined ones later
        self.usebounds = {'A': [0, 4 * self.startparam['A']],
                          'phi_0': [-360, 360],
                          'y_0': [0, self.startparam['y_0']]}

        if usebounds is not None:
            for thiskey in usebounds.keys():
                try:
                    self.usebounds[thiskey] = usebounds[thiskey]
                except:
                    print('WARNING: paramFITclass: the key in the usebounds is not valid : ' + thiskey)
                    print('         allowed ones:')
                    for tmp in self.usebounds.keys():
                        print('                ' + tmp)

        bounds = ((self.usebounds['A'][0], self.usebounds['phi_0'][0], self.usebounds['y_0'][0]),
                  (self.usebounds['A'][1], self.usebounds['phi_0'][1], self.usebounds['y_0'][1]))

        if printresults:
            print(p0)
            print(bounds)

        if varY is None:
            self.popt, self.pcov = curve_fit(SINEphase_func, X, Y, p0=p0, bounds=bounds)
        else:
            self.popt, self.pcov = curve_fit(SINEphase_func, X, Y, p0=p0, bounds=bounds, sigma=varY)

        self.fitres = {'A': self.popt[0], 'A_u': np.sqrt(self.pcov[0, 0]),
                       'phi_0': self.popt[1], 'phi_0_u': np.sqrt(self.pcov[1, 1]),
                       'y_0': self.popt[2], 'y_0_u': np.sqrt(self.pcov[2, 2])}

        if printresults:
            print("fit results:")
            for thiskey in self.fitres.keys():
                print("              %(key)s      : %(1)f " % {"key": thiskey, "1": self.fitres[thiskey]})

        # create the plot
        self.Xplot = np.linspace(X[0], X[-1], 10 * len(X))
        self.Yplot = SINEphase_func(self.Xplot, self.popt[0], self.popt[1], self.popt[2])

        return self.fitres, self.Xplot, self.Yplot


    def fit_ramsey(self, X,Y ,varY=None, usebounds = None, startparam = None, printresults=True):
        # X, Y (must have the same length)
        # varY is the variance of Y (used to weight the fit if given, must have the same length as X,Y)
        # startparam['A'] :  Amplitude
        # startparam['sigx'] : standard deviation
        # startparam['B'] : offset

        # usebounds['A'] : (min,max) of Amplitude
        # usebounds['sigx'] : (min,max) of standard deviation
        # usebounds['B'] : (min,max) of offset

        self.type = 'ramsey'
        self.X = X
        self.Y = Y
        self.varY = varY

        # define start parameters, if startparam is defined in the function then overwrite the defined ones later
        B_start = np.amin(Y)
        A_start = np.amax(Y) - B_start
        tmp = np.sum(Y > (B_start + A_start / 2))
        if tmp < 1:
            tmp = 1
        sigx_start = 0.5 * np.absolute((X[-1] - X[0]) / len(X)) * tmp
        self.startparam = {'A': A_start,
                           'sigx': sigx_start,
                           'B': B_start}

        if startparam is not None:
            for thiskey in startparam.keys():
                try:
                    self.startparam[thiskey] = startparam[thiskey]
                except:
                    print('WARNING: paramFITclass: the key in the startparam is not valid : ' + thiskey)
                    print('         allowed ones:')
                    for tmp in self.startparam.keys():
                        print('                ' + tmp)

        p0 = [self.startparam['A'], self.startparam['sigx'], self.startparam['B']]

        # define the bound parameters, if usebounds is defined in the function then overwrite the defined ones later
        self.usebounds = {'A': [0, 4 * self.startparam['A']],
                          'sigx': [0, X[-1]],
                          'B': [0, 2*np.amax(Y)]}

        if usebounds is not None:
            for thiskey in usebounds.keys():
                try:
                    self.usebounds[thiskey] = usebounds[thiskey]
                except:
                    print('WARNING: paramFITclass: the key in the usebounds is not valid : ' + thiskey)
                    print('         allowed ones:')
                    for tmp in self.usebounds.keys():
                        print('                ' + tmp)

        bounds = ((self.usebounds['A'][0], self.usebounds['sigx'][0], self.usebounds['B'][0]),
                  (self.usebounds['A'][1], self.usebounds['sigx'][1], self.usebounds['B'][1]))

        if printresults:
            print(p0)
            print(bounds)

        if varY is None:
            self.popt, self.pcov = curve_fit(ramsey_func, X, Y, p0=p0, bounds=bounds)
        else:
            self.popt, self.pcov = curve_fit(ramsey_func, X, Y, p0=p0, bounds=bounds, sigma = varY)

        self.fitres = {'A': self.popt[0],'A_u':np.sqrt(self.pcov[0, 0]),
                       'sigx': self.popt[1], 'sigx_u': np.sqrt(self.pcov[1, 1]),
                       'B': self.popt[2], 'B_u': np.sqrt(self.pcov[2, 2])}

        if printresults:
            print("fit results:")
            for thiskey in self.fitres.keys():
                print("              %(key)s      : %(1)f " % {"key":thiskey ,"1": self.fitres[thiskey]})

        self.Xplot = np.linspace(0,X[-1],10*len(X))
        self.Yplot = ramsey_func(self.Xplot, self.popt[0], self.popt[1], self.popt[2])

        return self.fitres, self.Xplot, self.Yplot

    def fit_ramsey_mod(self, X,Y ,varY=None, usebounds = None, startparam = None, printresults=True):
        # X, Y (must have the same length)
        # varY is the variance of Y (used to weight the fit if given, must have the same length as X,Y)
        # startparam['A'] :  Amplitude
        # startparam['sigx'] : standard deviation
        # startparam['B'] : offset

        # usebounds['A'] : (min,max) of Amplitude
        # usebounds['sigx'] : (min,max) of standard deviation
        # usebounds['B'] : (min,max) of offset

        self.type = 'ramsey'
        self.X = X
        self.Y = Y
        self.varY = varY

        # define start parameters, if startparam is defined in the function then overwrite the defined ones later
        B_start = np.amin(Y)
        A_start = np.amax(Y) - B_start
        tmp = np.sum(Y > (B_start + A_start / 2))
        if tmp < 1:
            tmp = 1
        sigx_start = 0.5 * np.absolute((X[-1] - X[0]) / len(X)) * tmp
        k_start = 0.1*A_start
        f0_start = 1/2

        self.startparam = {'A': A_start,
                           'sigx': sigx_start,
                           'k': k_start,
                           'f0': f0_start,
                           'B': B_start}

        if startparam is not None:
            for thiskey in startparam.keys():
                try:
                    self.startparam[thiskey] = startparam[thiskey]
                except:
                    print('WARNING: paramFITclass: the key in the startparam is not valid : ' + thiskey)
                    print('         allowed ones:')
                    for tmp in self.startparam.keys():
                        print('                ' + tmp)

        p0 = [self.startparam['A'], self.startparam['sigx'], self.startparam['k'], self.startparam['f0'], self.startparam['B']]

        # define the bound parameters, if usebounds is defined in the function then overwrite the defined ones later
        self.usebounds = {'A': [0, 4 * self.startparam['A']],
                          'sigx': [0, X[-1]],
                          'k': [0, 1],
                          'f0': [0, 20],
                          'B': [0, 2*np.amax(Y)]}

        if usebounds is not None:
            for thiskey in usebounds.keys():
                try:
                    self.usebounds[thiskey] = usebounds[thiskey]
                except:
                    print('WARNING: paramFITclass: the key in the usebounds is not valid : ' + thiskey)
                    print('         allowed ones:')
                    for tmp in self.usebounds.keys():
                        print('                ' + tmp)

        bounds = ((self.usebounds['A'][0], self.usebounds['sigx'][0], self.usebounds['k'][0], self.usebounds['f0'][0], self.usebounds['B'][0]),
                  (self.usebounds['A'][1], self.usebounds['sigx'][1], self.usebounds['k'][1], self.usebounds['f0'][1], self.usebounds['B'][1]))

        if printresults:
            print(p0)
            print(bounds)

        if varY is None:
            self.popt, self.pcov = curve_fit(ramsey_mod_func, X, Y, p0=p0, bounds=bounds)
        else:
            self.popt, self.pcov = curve_fit(ramsey_mod_func, X, Y, p0=p0, bounds=bounds, sigma = varY)

        self.fitres = {'A': self.popt[0],'A_u':np.sqrt(self.pcov[0, 0]),
                       'sigx': self.popt[1], 'sigx_u': np.sqrt(self.pcov[1, 1]),
                       'k': self.popt[2], 'k_u': np.sqrt(self.pcov[2, 2]),
                       'f0': self.popt[3], 'f0_u': np.sqrt(self.pcov[3, 3]),
                       'B': self.popt[4], 'B_u': np.sqrt(self.pcov[4, 4])}

        if printresults:
            print("fit results:")
            for thiskey in self.fitres.keys():
                print("              %(key)s      : %(1)f " % {"key":thiskey ,"1": self.fitres[thiskey]})

        self.Xplot = np.linspace(0,X[-1],10*len(X))
        self.Yplot = ramsey_mod_func(self.Xplot, self.popt[0], self.popt[1], self.popt[2], self.popt[3], self.popt[4])

        return self.fitres, self.Xplot, self.Yplot, ramsey_func(self.Xplot, self.popt[0], self.popt[1], self.popt[4])

    def fit_hahn_echo(self, X,Y ,varY=None, usebounds = None, startparam = None, printresults=True):
        # X, Y (must have the same length)
        # varY is the variance of Y (used to weight the fit if given, must have the same length as X,Y)
        # startparam['A'] :  Amplitude
        # startparam['sigx'] : standard deviation
        # startparam['B'] : offset

        # usebounds['A'] : (min,max) of Amplitude
        # usebounds['sigx'] : (min,max) of standard deviation
        # usebounds['B'] : (min,max) of offset

        self.type = 'ramsey'
        self.X = X
        self.Y = Y
        self.varY = varY

        # hahn_echo_func(x, A, T2, n, k, f0, f1)
        # define start parameters, if startparam is defined in the function then overwrite the defined ones later

        A_start = np.amax(Y) - np.amin(Y)
        tmp = np.sum(Y > (A_start / 2))
        if tmp < 1:
            tmp = 1
        T2_start = 0.5 * np.absolute((X[-1] - X[0]) / len(X)) * tmp
        n_start = 1
        k_start = 0.8*A_start
        f0_start = 170*0.001/2
        self.startparam = {'A': A_start,
                           'T2': T2_start,
                           'n': n_start,
                           'k': k_start,
                           'f0': f0_start}


        if startparam is not None:
            for thiskey in startparam.keys():
                try:
                    self.startparam[thiskey] = startparam[thiskey]
                except:
                    print('WARNING: paramFITclass: the key in the startparam is not valid : ' + thiskey)
                    print('         allowed ones:')
                    for tmp in self.startparam.keys():
                        print('                ' + tmp)

        p0 = [self.startparam['A'], self.startparam['T2'], self.startparam['n'], self.startparam['k'], self.startparam['f0']]

        # define the bound parameters, if usebounds is defined in the function then overwrite the defined ones later
        self.usebounds = {'A': [0, 4 * self.startparam['A']],
                          'T2': [0, X[-1]],
                          'n': [1, 3],
                          'k': [0, 1],
                          'f0': [0, 20]}


        if usebounds is not None:
            for thiskey in usebounds.keys():
                try:
                    self.usebounds[thiskey] = usebounds[thiskey]
                except:
                    print('WARNING: paramFITclass: the key in the usebounds is not valid : ' + thiskey)
                    print('         allowed ones:')
                    for tmp in self.usebounds.keys():
                        print('                ' + tmp)

        bounds = ((self.usebounds['A'][0], self.usebounds['T2'][0], self.usebounds['n'][0], self.usebounds['k'][0], self.usebounds['f0'][0]),
                  (self.usebounds['A'][1], self.usebounds['T2'][1], self.usebounds['n'][1], self.usebounds['k'][1], self.usebounds['f0'][1]))

        if printresults:
            print(p0)
            print(bounds)

        if varY is None:
            self.popt, self.pcov = curve_fit(hahn_echo_func, X, Y, p0=p0, bounds=bounds)
        else:
            self.popt, self.pcov = curve_fit(hahn_echo_func, X, Y, p0=p0, bounds=bounds, sigma = varY)

        self.fitres = {'A': self.popt[0],'A_u':np.sqrt(self.pcov[0, 0]),
                       'T2': self.popt[1], 'T2_u': np.sqrt(self.pcov[1, 1]),
                       'n': self.popt[2], 'n_u': np.sqrt(self.pcov[2, 2]),
                       'k': self.popt[3], 'k_u': np.sqrt(self.pcov[3, 3]),
                       'f0': self.popt[4], 'f0_u': np.sqrt(self.pcov[4, 4])}

        if printresults:
            print("fit results:")
            for thiskey in self.fitres.keys():
                print("              %(key)s      : %(1)f " % {"key":thiskey ,"1": self.fitres[thiskey]})

        self.Xplot = np.linspace(0,X[-1],10*len(X))
        self.Yplot = hahn_echo_func(self.Xplot, self.popt[0], self.popt[1], self.popt[2], self.popt[3], self.popt[4])

        return self.fitres, self.Xplot, self.Yplot


    # def fit_hahn_echo(self, X,Y ,varY=None, usebounds = None, startparam = None, printresults=True):
    #     # X, Y (must have the same length)
    #     # varY is the variance of Y (used to weight the fit if given, must have the same length as X,Y)
    #     # startparam['A'] :  Amplitude
    #     # startparam['sigx'] : standard deviation
    #     # startparam['B'] : offset
    #
    #     # usebounds['A'] : (min,max) of Amplitude
    #     # usebounds['sigx'] : (min,max) of standard deviation
    #     # usebounds['B'] : (min,max) of offset
    #
    #     self.type = 'ramsey'
    #     self.X = X
    #     self.Y = Y
    #     self.varY = varY
    #
    #     # hahn_echo_func(x, A, T2, n, k, f0, f1)
    #     # define start parameters, if startparam is defined in the function then overwrite the defined ones later
    #
    #     A_start = np.amax(Y) - np.amin(Y)
    #     tmp = np.sum(Y > (A_start / 2))
    #     if tmp < 1:
    #         tmp = 1
    #     T2_start = 0.5 * np.absolute((X[-1] - X[0]) / len(X)) * tmp
    #     n_start = 2
    #     k_start = 0.1*A_start
    #     f0_start = 170*0.001
    #     f1_start = 1 + f0_start
    #     self.startparam = {'A': A_start,
    #                        'T2': T2_start,
    #                        'n': n_start,
    #                        'k': k_start,
    #                        'f0': f0_start,
    #                        'f1': f1_start}
    #
    #
    #     if startparam is not None:
    #         for thiskey in startparam.keys():
    #             try:
    #                 self.startparam[thiskey] = startparam[thiskey]
    #             except:
    #                 print('WARNING: paramFITclass: the key in the startparam is not valid : ' + thiskey)
    #                 print('         allowed ones:')
    #                 for tmp in self.startparam.keys():
    #                     print('                ' + tmp)
    #
    #     p0 = [self.startparam['A'], self.startparam['T2'], self.startparam['n'], self.startparam['k'], self.startparam['f0'], self.startparam['f1']]
    #
    #     # define the bound parameters, if usebounds is defined in the function then overwrite the defined ones later
    #     self.usebounds = {'A': [0, 4 * self.startparam['A']],
    #                       'T2': [0, X[-1]],
    #                       'n': [1, 3],
    #                       'k': [0, 1],
    #                       'f0': [0, 1000],
    #                       'f1': [0, 1000]}
    #
    #
    #     if usebounds is not None:
    #         for thiskey in usebounds.keys():
    #             try:
    #                 self.usebounds[thiskey] = usebounds[thiskey]
    #             except:
    #                 print('WARNING: paramFITclass: the key in the usebounds is not valid : ' + thiskey)
    #                 print('         allowed ones:')
    #                 for tmp in self.usebounds.keys():
    #                     print('                ' + tmp)
    #
    #     bounds = ((self.usebounds['A'][0], self.usebounds['T2'][0], self.usebounds['n'][0], self.usebounds['k'][0], self.usebounds['f0'][0], self.usebounds['f1'][0]),
    #               (self.usebounds['A'][1], self.usebounds['T2'][1], self.usebounds['n'][1], self.usebounds['k'][1], self.usebounds['f0'][1], self.usebounds['f1'][1]))
    #
    #     if printresults:
    #         print(p0)
    #         print(bounds)
    #
    #     if varY is None:
    #         self.popt, self.pcov = curve_fit(hahn_echo_func, X, Y, p0=p0, bounds=bounds)
    #     else:
    #         self.popt, self.pcov = curve_fit(hahn_echo_func, X, Y, p0=p0, bounds=bounds, sigma = varY)
    #
    #     self.fitres = {'A': self.popt[0],'A_u':np.sqrt(self.pcov[0, 0]),
    #                    'T2': self.popt[1], 'T2_u': np.sqrt(self.pcov[1, 1]),
    #                    'n': self.popt[2], 'n_u': np.sqrt(self.pcov[2, 2]),
    #                    'k': self.popt[3], 'k_u': np.sqrt(self.pcov[3, 3]),
    #                    'f0': self.popt[4], 'f0_u': np.sqrt(self.pcov[4, 4]),
    #                    'f1': self.popt[5], 'f1_u': np.sqrt(self.pcov[5, 5])}
    #
    #     if printresults:
    #         print("fit results:")
    #         for thiskey in self.fitres.keys():
    #             print("              %(key)s      : %(1)f " % {"key":thiskey ,"1": self.fitres[thiskey]})
    #
    #     self.Xplot = np.linspace(0,X[-1],10*len(X))
    #     self.Yplot = hahn_echo_func(self.Xplot, self.popt[0], self.popt[1], self.popt[2], self.popt[3], self.popt[4], self.popt[5])
    #
    #     return self.fitres, self.Xplot, self.Yplot