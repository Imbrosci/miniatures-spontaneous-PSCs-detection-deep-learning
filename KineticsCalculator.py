# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:22:29 2022.

@author: barbara
"""


import numpy as np
from scipy.optimize import curve_fit
import warnings


class PSCsKinetics():
    """Calculate the kinetics from the mean of the detected PSCs."""

    def __init__(self, signal_lp, x_dp_final, fs):
        self.signal_lp = signal_lp
        self.x_dp_final = x_dp_final
        self.fs = fs
        self.rise_curve_rise_time()
        self.decay_curve_decay_time()

    def rise_fit(self, x, a, b, c):
        """Provide the function to fit the rise phase of the PSCs."""
        warnings.filterwarnings('ignore')
        return a * np.exp(b * x) + c

    def decay_fit(self, x, a, b, c):
        """Provide the function to fit the decay phase of the PSCs."""
        warnings.filterwarnings('ignore')
        return a * np.exp(-b * x) + c

    def rise_curve_rise_time(self):
        """
        Fit the rise phase and calculate the 90-10% rise time.

        The average of the detected PSCs are used for fitting. Cutouts of PSCs
        (20 ms to the through) are used to calculate the average signal. PSCs
        close to the beginning of the recording (<20ms) are excluded from
        calculating the average signal.
        """
        cutouts_rise = []
        ms_per_dp = 1 / (self.fs / 1000)
        x_dp_final = self.x_dp_final
        for i in range(len(x_dp_final)):

            # create cutouts from the rise phase
            steps = int(self.fs / 1000 * 20)
            stop = 20 - ms_per_dp
            xdata_rise = np.linspace(0, stop, steps)
            delta_left = min(steps, x_dp_final[i])
            ydata = self.signal_lp[x_dp_final[i] - delta_left:
                                   x_dp_final[i]].reshape(-1,)

            if len(ydata) == steps:
                cutouts_rise.append(ydata)

        cutout_rise_mean = np.mean(np.array(cutouts_rise), axis=0)

        # try to fit the rise phase and to calculate the rise time
        try:
            popt, pcov = curve_fit(self.rise_fit, xdata_rise, cutout_rise_mean)
            fcr = self.rise_fit(xdata_rise, *popt)

            # get the absolute value and find the range values
            fcr_abs = abs(fcr)
            fcr_range = np.max(fcr_abs) - np.min(fcr_abs)
            # normalize ydata_new between 0 and 1
            fcr_abs_scaled = (fcr_abs - np.min(fcr_abs)) / fcr_range

            # get dp within 10-90 %
            rise_time_dp = np.sum(np.histogram(
                fcr_abs_scaled, bins=100)[0][10:90])

            # calculate the rise time (10-90 %)
            rise_time = rise_time_dp / (self.fs / 1000)

        except (ValueError, RuntimeError):
            fcr = np.nan
            rise_time = np.nan

        return fcr, rise_time

    def decay_curve_decay_time(self):
        """
        Fit the decay phase and calculate the 90-10% decay time.

        The average of the detected PSCs are used for fitting. Cutouts of PSCs
        (50 ms starting from the through) are used to calculate the average
        signal. PSCs close to the end of the recording (<50ms) are excluded
        from calculating the average signal.
        """
        cutouts_decay = []
        ms_per_dp = 1 / (self.fs / 1000)
        x_dp_final = self.x_dp_final
        for i in range(len(x_dp_final)):

            # create cutouts from the decay phase
            steps = int(self.fs / 1000 * 50)
            stop = 50 - ms_per_dp
            xdata_decay = np.linspace(0, stop, steps)
            delta_right = min(steps, self.signal_lp.shape[0] - (x_dp_final[i]))
            ydata = self.signal_lp[x_dp_final[i]:
                                   x_dp_final[i] + delta_right].reshape(-1,)

            if len(ydata) == steps:
                cutouts_decay.append(ydata)

        cutout_decay_mean = np.mean(np.array(cutouts_decay), axis=0)

        # try to fit the decay phase and to calculate the decay time
        try:
            popt, pcov = curve_fit(self.decay_fit, xdata_decay,
                                   cutout_decay_mean)
            fcd = self.decay_fit(xdata_decay, *popt)

            # get the absolute value and find the range values
            fcd_abs = abs(fcd)
            fcd_range = np.max(fcd_abs) - np.min(fcd_abs)
            # normalize ydata_new between 0 and 1
            fcd_abs_scaled = (fcd_abs - np.min(fcd_abs)) / fcd_range

            # get dp within 10-90 %
            decay_time_dp = np.sum(np.histogram(
                fcd_abs_scaled, bins=100)[0][10:90])

            # calculate the decay time (10-90 %)
            decay_time = decay_time_dp / (self.fs / 1000)
        except (ValueError, RuntimeError):
            fcd = np.nan
            decay_time = np.nan

        return fcd, decay_time
